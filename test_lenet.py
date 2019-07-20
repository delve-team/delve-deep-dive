import torch
from torch.utils.data import SubsetRandomSampler, DataLoader
from torch import nn, optim

from datasets import Food101Dataset
from models import LeNetModel


batch_size = 10
criterion = nn.CrossEntropyLoss()
net = LeNetModel()
optimizer = optim.SGD(net.parameters(), lr=0.001)
data = Food101Dataset('tmp')
data.init(selected_classes=["foie_gras"])

TrainingDataLoader = DataLoader(data, batch_size=batch_size, sampler=SubsetRandomSampler(data.training_indices()))
TestDataLoader = DataLoader(data, batch_size=batch_size, sampler=SubsetRandomSampler(data.test_indices()))


for index, data in enumerate(TrainingDataLoader):
    batch_out = net(data)

