import os
from imageio import imread
from PIL import Image
import tarfile
from urllib.request import urlretrieve
from collections import OrderedDict
from torch.utils.data import SubsetRandomSampler, DataLoader
from torch.utils.data import Dataset
import json
import scipy
import scipy.misc
import numpy as np

import torch.utils.data

import torchvision
from torchvision import transforms

func_template = """
def Cifar10_{}(batch_size=12, output_size=(224, 224), cache_dir='tmp'):

    # Transformations
    RC = transforms.RandomCrop((32, 32), padding=4)
    RHF = transforms.RandomHorizontalFlip()
    RVF = transforms.RandomVerticalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    TPIL = transforms.ToPILImage()
    RS = transforms.Resize({})

    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([RC, RHF, RS, TT, NRM])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([RS, TT, NRM])


    trainset = torchvision.datasets.CIFAR10(root=cache_dir, train=True,
                                            download=True, transform=transform_with_aug)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=8, pin_memory=True)
    testset = torchvision.datasets.CIFAR10(root=cache_dir, train=False,
                                           download=True, transform=transform_no_aug)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=8, pin_memory=True)
    train_loader.name = "Cifar10_{}"
    return train_loader, test_loader, ({}, {}), 10
    
    """

resolution = [38, 75, 78, 88, 108, 109, 132, 140, 150, 238, 264, 300, 435, 600]

for res in resolution:
    func_code = func_template.format(res, res, res, res, res)
    exec(func_code)

