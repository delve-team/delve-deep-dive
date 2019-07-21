import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor
from functools import partial, reduce
from operator import mul

class LeNetModel(nn.Module):
    name = "LeNet"

    @staticmethod
    def _input_fc_size(input_size: int):
        conv_size1 = input_size - 2
        pool_size1 = floor((conv_size1 - 2) / 2) + 1
        conv_size2 = pool_size1 - 2
        pool_size2 = floor((conv_size2 - 2) / 2) + 1
        return pool_size2

    def __init__(self, input_size=(512,512), output_size=2):
        super(LeNetModel, self).__init__()
        self.input_size = input_size
        self.input_fc_dims = tuple(map(self._input_fc_size, input_size))
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * mul(*self.input_fc_dims), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x.float())))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * mul(*self.input_fc_dims))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

