import torch
import torchvision
from torch import nn, optim
from torch.utils.data import SubsetRandomSampler, DataLoader
import torchvision.transforms as transforms

from datasets import Food101Dataset
from models import LeNetModel
from trainer import Trainer


batch_size = 10
criterion = nn.CrossEntropyLoss()
model = LeNetModel((512, 512), 2)
#optimizer = optim.SGD(net.parameters(), lr=0.001)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



def get_food101_dataset(cache_dir='tmp', selected_classes=list()):
    dataset = Food101Dataset(cache_dir)
    dataset.init(selected_classes=selected_classes)
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=SubsetRandomSampler(dataset.training_indices()))
    test_loader = DataLoader(dataset, batch_size=batch_size,
                             sampler=SubsetRandomSampler(dataset.test_indices()))
    train_loader.name = "food101"
    test_loader.name = "food101"
    return train_loader, test_loader

def get_cifar10_dataset(cache_dir='tmp'):
    trainset = torchvision.datasets.CIFAR10(root=cache_dir, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root=cache_dir, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    trainloader.name = "cifar10"
    return trainloader, testloader

selected_classes = ['foie_gras', 'tacos']
train_loader, test_loader = get_food101_dataset('tmp',selected_classes)
#train_loader, test_loader = get_cifar10_dataset()

trainer = Trainer(model, train_loader, test_loader)
trainer.train()
trainer.test()

#TrainingDataLoader = DataLoader(data, batch_size=batch_size, sampler=SubsetRandomSampler(data.training_indices()))
#TestDataLoader = DataLoader(data, batch_size=batch_size, sampler=SubsetRandomSampler(data.test_indices()))


#for index, data in enumerate(TrainingDataLoader):
#    batch_out = net(data)

