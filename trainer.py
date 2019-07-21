import torch
import torch.optim as optim
import torch.nn as nn
from delve.torchcallback import CheckLayerSat
import os
from datetime import datetime

def now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

class Trainer:

    def __init__(self, model, train_loader, test_loader, epochs=200, batch_size=60, logs_dir='logs'):
        self.model = model
        self.epochs = epochs


        self.train_loader = train_loader
        self.test_loader = test_loader

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        save_dir = os.path.join(logs_dir, model.name, train_loader.name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.stats = CheckLayerSat(os.path.join(save_dir, str(batch_size)), 'csv', model, stats=['lsat'])

    def train(self):
        for epoch in range(self.epochs):
            print("{} Epoch {}, loss: {}, accuracy: {}".format(now(), epoch, *self.train_epoch()))
            self.test()
            self.stats.add_saturations()
            self.stats.save()
        self.stats.close()

    def train_epoch(self):
        self.model.train()
        correct = 0
        total = 0
        running_loss = 0
        for batch, data in enumerate(self.train_loader):
            inputs, labels = data

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
        self.stats.add_scalar('training_loss', running_loss)
        self.stats.add_scalar('training_accuracy', correct/total)
        return running_loss, correct/total

    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        test_loss = 0
        with torch.no_grad():
            for batch, data in enumerate(self.test_loader):
                inputs, labels = data
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                test_loss += loss.item()
        self.stats.add_scalar('test_loss', test_loss)
        self.stats.add_scalar('test_accuracy', correct/total)
        print('{} Accuracy on {} images: {:.2f}'.format(now(), total, correct/total))
