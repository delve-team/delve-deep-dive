import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from delve.torchcallback import CheckLayerSat
import os
from datetime import datetime
import pandas as pd
from radam import RAdam

def now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
class Trainer:

    def __init__(self, model, train_loader, test_loader, epochs=200, batch_size=60, run_id=0, logs_dir='logs', device='cpu', optimizer='None'):
        self.device = device
        self.model = model
        self.epochs = epochs

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

        savepath = os.path.join(save_dir, f'{model.name}_bs{batch_size}_e{epochs}_id{run_id}.csv')
        self.experiment_done = False
        if os.path.exists(savepath):
            trained_epochs = len(pd.read_csv(savepath, sep=';'))

            if trained_epochs >= epochs:
                self.experiment_done = True
                print('Experiment Logs for the exact same experiment with identical run_id was detecting, training will be skipped, consider using another run_id')



        self.stats = CheckLayerSat(os.path.join(save_dir, f'{model.name}_bs{batch_size}_e{epochs}_id{run_id}'), 'csv', model, stats=['lsat'], sat_threshold=.99, verbose=False, conv_method='mean', log_interval=1, device=device)

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
            self.stats.save()
        self.stats.close()

    def train_epoch(self):
        self.model.train()
        correct = 0
        total = 0
        running_loss = 0
        for batch, data in enumerate(self.train_loader):
            if batch%100 == 0:
                print(batch, 'of', len(self.train_loader))
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            correct += (predicted == labels).sum().item()
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
        self.stats.add_scalar('training_loss', running_loss/total)
        self.stats.add_scalar('training_accuracy', correct/total)
        return running_loss/total, correct/total

    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        test_loss = 0
        with torch.no_grad():
            for batch, data in enumerate(self.test_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                test_loss += loss.item()
        self.stats.add_scalar('test_loss', test_loss/total)
        self.stats.add_scalar('test_accuracy', correct/total)
        print('{} Test Accuracy on {} images: {:.2f}'.format(now(), total, correct/total))
