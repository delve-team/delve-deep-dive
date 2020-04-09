import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from delve.torchcallback import CheckLayerSat
from delve.writers import CSVandPlottingWriter, NPYWriter
import os
from datetime import datetime
import pandas as pd
from radam import RAdam
from time import time
from pca_layers import LinearPCALayer, Conv2DPCALayer
from pca_layers import change_all_pca_layer_thresholds
from saturation_plotter import plot_saturation_level_from_results


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred).long())

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


def now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


class Trainer:

    def __init__(self, model,
                 train_loader,
                 test_loader,
                 epochs=200,
                 batch_size=60,
                 run_id=0,
                 logs_dir='logs',
                 device='cpu',
                 saturation_device=None,
                 optimizer='None',
                 plot=True,
                 compute_top_k=False,
                 data_prallel=False,
                 conv_method='channelwise',
                 thresh=.99,
                 half_precision=False,
                 downsampling=None):
        self.saturation_device = device if saturation_device is None else saturation_device
        self.device = device
        self.model = model
        self.epochs = epochs
        self.plot = plot
        self.compute_top_k = compute_top_k

        if 'cuda' in device:
            cudnn.benchmark = True

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.criterion = nn.CrossEntropyLoss()
        print('Checking for optimizer for {}'.format(optimizer))
        #optimizer = str(optimizer)
        if optimizer == "adam":
            print('Using adam')
            self.optimizer = optim.Adam(model.parameters())
        elif optimizer == 'bad_lr_adam':
            print('Using adam with to large learning rate')
            self.optimizer = optim.Adam(model.parameters(), lr=0.01)
        elif optimizer == "SGD":
            print('Using SGD')
            self.optimizer = optim.SGD(model.parameters(), lr=0.0, momentum=0.9)
        elif optimizer == "LRS":
            print('Using LRS')
            self.optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, 5)
        elif optimizer == "radam":
            print('Using radam')
            self.optimizer = RAdam(model.parameters())
        else:
            raise ValueError('Unknown optimizer {}'.format(optimizer))
        self.opt_name = optimizer
        save_dir = os.path.join(logs_dir, model.name, train_loader.name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.savepath = os.path.join(save_dir, f'{model.name}_bs{batch_size}_e{epochs}_dspl{downsampling}_t{int(thresh*1000)}_id{run_id}.csv')
        self.experiment_done = False
        if os.path.exists(self.savepath):
            trained_epochs = len(pd.read_csv(self.savepath, sep=';'))

            if trained_epochs >= epochs:
                self.experiment_done = True
                print(f'Experiment Logs for the exact same experiment with identical run_id was detected, training will be skipped, consider using another run_id')
        if os.path.exists((self.savepath.replace('.csv', '.pt'))):
            self.model.load_state_dict(torch.load(self.savepath.replace('.csv', '.pt'))['model_state_dict'])
            if data_prallel:
                self.model = nn.DataParallel(self.model)
            else:
                self.model = self.model.to(self.device)
            if half_precision:
                self.model = self.model.half()
            self.optimizer.load_state_dict(torch.load(self.savepath.replace('.csv', '.pt'))['optimizer'])
            self.start_epoch = torch.load(self.savepath.replace('.csv', '.pt'))['epoch'] + 1
            initial_epoch = self._infer_initial_epoch(self.savepath)
            print('Resuming existing run, starting at epoch', self.start_epoch, 'from', self.savepath.replace('.csv', '.pt'))
        else:
            if half_precision:
                self.model = self.model.half()
            self.start_epoch = 0
            initial_epoch = 0
            self.parallel = data_prallel
            if data_prallel:
                self.model = nn.DataParallel(self.model)
            else:
                self.model = self.model.to(self.device)
        writer = CSVandPlottingWriter(self.savepath.replace('.csv', ''), fontsize=16, primary_metric='test_accuracy')
        writer2 = NPYWriter(self.savepath.replace('.csv', ''))
        self.pooling_strat = conv_method
        print('Settomg Satiraton recording threshold to', thresh)
        self.half = half_precision

        self.stats = CheckLayerSat(self.savepath.replace('.csv', ''),
                                   [writer],
                                   model, ignore_layer_names='convolution',
                                   stats=['lsat', 'idim'],
                                   sat_threshold=.99, verbose=False,
                                   conv_method=conv_method, log_interval=1,
                                   device=self.saturation_device, reset_covariance=True,
                                   max_samples=None, initial_epoch=initial_epoch, interpolation_strategy='nearest' if downsampling is not None else None,
                                   interpolation_downsampling=downsampling)


    def _infer_initial_epoch(self, savepath):
        if not os.path.exists(savepath):
            return 0
        else:
            df = pd.read_csv(savepath, sep=';', index_col=0)
            print(len(df)+1)
            return len(df)

    def train(self):
        if self.experiment_done:
            return
        for epoch in range(self.start_epoch, self.epochs):
            #self.test(epoch=epoch)

            print('Start training epoch', epoch)
            print("{} Epoch {}, training loss: {}, training accuracy: {}".format(now(), epoch, *self.train_epoch()))
            self.test(epoch=epoch)
            if self.opt_name == "LRS":
                print('LRS step')
                self.lr_scheduler.step() 
            self.stats.add_saturations()
            #self.stats.save()
            #if self.plot:
            #    plot_saturation_level_from_results(self.savepath, epoch)
        self.stats.close()
        return self.savepath+'.csv'

    def train_epoch(self):
        self.model.train()
        correct = 0
        total = 0
        running_loss = 0
        old_time = time()
        top5_accumulator = 0
        for batch, data in enumerate(self.train_loader):
            if batch%10 == 0 and batch != 0:
                print(batch, 'of', len(self.train_loader), 'processing time', time()-old_time, "top5_acc:" if self.compute_top_k else 'acc:', round(top5_accumulator/(batch),3) if self.compute_top_k else correct/total)
                old_time = time()
            inputs, labels = data
            if self.half:
                inputs, labels = inputs.to(self.device).half(), labels.to(self.device)
            else:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            if self.compute_top_k:
                top5_accumulator += accuracy(outputs, labels, (5,))[0]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            correct += (predicted == labels.long()).sum().item()

            running_loss += loss.item()
        self.stats.add_scalar('training_loss', running_loss/total)
        if self.compute_top_k:
            self.stats.add_scalar('training_accuracy', (top5_accumulator/(batch+1)))
        else:
            self.stats.add_scalar('training_accuracy', correct/total)
        return running_loss/total, correct/total

    def test(self, epoch, save=True):
        self.model.eval()
        correct = 0
        total = 0
        test_loss = 0
        top5_accumulator = 0
        with torch.no_grad():
            for batch, data in enumerate(self.test_loader):
                if batch%10 == 0:
                    print('Processing eval batch', batch,'of', len(self.test_loader))
                inputs, labels = data
                if self.half:
                    inputs, labels = inputs.to(self.device).half(), labels.to(self.device)
                else:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.long()).sum().item()
                if self.compute_top_k:
                    top5_accumulator += accuracy(outputs, labels, (5,))[0]
                test_loss += loss.item()

        self.stats.add_scalar('test_loss', test_loss/total)
        if self.compute_top_k:
            self.stats.add_scalar('test_accuracy', top5_accumulator/(batch+1))
            print('{} Test Top5-Accuracy on {} images: {:.4f}'.format(now(), total, top5_accumulator/(batch+1)))

        else:
            self.stats.add_scalar('test_accuracy', correct/total)
            print('{} Test Accuracy on {} images: {:.4f}'.format(now(), total, correct/total))
        if save:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': epoch,
                'test_loss': test_loss / total
            }, self.savepath.replace('.csv', '.pt'))
        return correct / total, test_loss / total

