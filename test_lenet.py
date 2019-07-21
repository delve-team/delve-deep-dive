import torch
import torchvision
from torch import nn, optim
import torchvision.transforms as transforms
from datasets import get_food101_dataset, get_cifar10_dataset
from models import LeNetModel, vgg13
from trainer import Trainer


batch_size = 10
criterion = nn.CrossEntropyLoss()
#model = LeNetModel((512, 512), 2)
model = vgg13()


selected_classes = ['foie_gras', 'tacos']
train_loader, test_loader = get_food101_dataset(selected_classes=selected_classes)
#train_loader, test_loader = get_cifar10_dataset()

trainer = Trainer(model, train_loader, test_loader)
trainer.train()
trainer.test()

#TrainingDataLoader = DataLoader(data, batch_size=batch_size, sampler=SubsetRandomSampler(data.training_indices()))
#TestDataLoader = DataLoader(data, batch_size=batch_size, sampler=SubsetRandomSampler(data.test_indices()))


#for index, data in enumerate(TrainingDataLoader):
#    batch_out = net(data)

