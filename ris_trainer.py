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
                 conv_method='mean'):
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

        self.criterion = nn.MSELoss()
        print('Checking for optimizer for {}'.format(optimizer))
        if optimizer == "adam":
            print('Using adam')
            self.optimizer = optim.Adam(model.parameters())
        elif optimizer == "SGD":
            print('Using SGD')
            self.optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
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

        self.savepath = os.path.join(save_dir, f'{model.name}_bs{batch_size}_e{epochs}_id{run_id}.csv')
        self.experiment_done = False
        if os.path.exists(self.savepath):
            trained_epochs = len(pd.read_csv(self.savepath, sep=';'))

            if trained_epochs >= epochs:
                self.experiment_done = True
                print(f'Experiment Logs for the exact same experiment {self.savepath} with identical run_id was detecting, training will be skipped, consider using another run_id')
        self.parallel = data_prallel
        if data_prallel:
            self.model = nn.DataParallel(self.model, ['cuda:0', 'cuda:1'])
        writer = CSVandPlottingWriter(self.savepath.replace('.csv', ''), fontsize=16, primary_metric='test_loss')
        self.pooling_strat = conv_method
        self.stats = CheckLayerSat(self.savepath.replace('.csv', ''), writer, model, stats=['lsat'], sat_threshold=.99, verbose=False, conv_method=conv_method, log_interval=1, device=self.saturation_device, reset_covariance=True, max_samples=None, ignore_layer_names='classifier666')

    def train(self):
        if self.experiment_done:
            return
        self.model.to(self.device)
        for epoch in range(self.epochs):
            print("{} Epoch {}, training loss: {}".format(now(), epoch, self.train_epoch()))
            torch.save({'model_state_dict': self.model.state_dict()}, self.savepath.replace('.csv', '.pt'))
            self.test()
            if self.opt_name == "LRS":
                print('LRS step')
                self.lr_scheduler.step()
            self.stats.add_saturations()
            #    plot_saturation_level_from_results(self.savepath, epoch)
        self.stats.close()
        return self.savepath+'.csv'

    def train_epoch(self):
        self.model.train()
        total = 0
        running_loss = 0
        old_time = time()
        for batch, data in enumerate(self.train_loader):
            if batch%5 == 0 and batch != 0:
                print(batch, 'of', len(self.train_loader), 'processing time', time()-old_time, 'loss:', running_loss/total)
                old_time = time()
            inputs, _ = data
            inputs = inputs.to(self.device)
            total += inputs.size(0)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            loss = self.criterion(outputs, inputs)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
        self.stats.add_scalar('training_loss', running_loss/total)
        return running_loss/total

    def test(self):
        self.model.eval()
        total = 0
        test_loss = 0
        with torch.no_grad():
            for batch, data in enumerate(self.test_loader):
                inputs, _ = data
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_loss += loss.item()
                total += inputs.size(0)

        self.stats.add_scalar('test_loss', test_loss/total)
        print('{} Test Loss on {} images: {:.2f}'.format(now(), total, test_loss/total))
        torch.save({'model_state_dict': self.model.state_dict()}, self.savepath.replace('.csv', '.pt'))
