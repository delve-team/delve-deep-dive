import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from delve.torchcallback import CheckLayerSat
from delve.writers import CSVandPlottingWriter
import os
from datetime import datetime
import pandas as pd
from radam import RAdam
from time import time
from pca_layers import change_all_pca_layer_thresholds
from saturation_plotter import plot_saturation_level_from_results

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

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
                 thresh=.99):
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
            self.optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.9)
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

        self.savepath = os.path.join(save_dir, f'{model.name}_bs{batch_size}_e{epochs}_t{int(thresh*1000)}_id{run_id}.csv')
        self.experiment_done = False
        if os.path.exists(self.savepath):
            trained_epochs = len(pd.read_csv(self.savepath, sep=';'))

            if trained_epochs >= epochs:
                self.experiment_done = True
                print(f'Experiment Logs for the exact same experiment with identical run_id was detecting, training will be skipped, consider using another run_id')
        self.parallel = data_prallel
        if data_prallel:
            self.model = nn.DataParallel(self.model, ['cuda:0', 'cuda:1'])
        writer = CSVandPlottingWriter(self.savepath.replace('.csv', ''), fontsize=16, primary_metric='test_accuracy')
        self.pooling_strat = conv_method
        print('Settomg Satiraton recording threshold to', thresh)
        self.stats = CheckLayerSat(self.savepath.replace('.csv', ''), writer, model, ignore_layer_names='convolution', stats=['lsat'], sat_threshold=.99, verbose=False, conv_method=conv_method, log_interval=1, device=self.saturation_device, reset_covariance=True, max_samples=None)

    def train(self):
        if self.experiment_done:
            return
        self.model.to(self.device)
        for epoch in range(self.epochs):
            print("{} Epoch {}, training loss: {}, training accuracy: {}".format(now(), epoch, *self.train_epoch()))
            self.test()
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
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            if self.compute_top_k:
                top5_accumulator += accuracy(outputs, labels, (5,))[0]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            correct += (predicted == labels).sum().item()
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
        self.stats.add_scalar('training_loss', running_loss/total)
        if self.compute_top_k:
            self.stats.add_scalar('training_accuracy', (top5_accumulator/(batch+1)))
        else:
            self.stats.add_scalar('training_accuracy', correct/total)
        return running_loss/total, correct/total

    def test(self, save=True):
        self.model.eval()
        correct = 0
        total = 0
        test_loss = 0
        top5_accumulator = 0
        with torch.no_grad():
            for batch, data in enumerate(self.test_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
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
            torch.save({'model_state_dict': self.model.state_dict()}, self.savepath.replace('.csv', '.pt'))
        return correct / total, test_loss / total

