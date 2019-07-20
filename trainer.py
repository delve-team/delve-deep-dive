import torch
import torch.optim as optim
import torch.nn as nn


class Trainer:

    def __init__(self, model, train_loader, test_loader, epochs=10, batch_size=12):
        self.model = model
        self.epochs = epochs


        self.train_loader = train_loader
        self.test_loader = test_loader

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            print("Epoch {}, loss: {}".format(epoch, self.train_epoch()))

    def train_epoch(self):
        running_loss = 0
        for batch, data in enumerate(self.train_loader):
            inputs, labels = data

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
        return running_loss

    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch, data in enumerate(self.test_loader):
                inputs, labels = data
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy on {} images: {:.2f}'.format(total, correct/total))
