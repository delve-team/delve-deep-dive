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
from exp_datasets.special_cifar10 import *
from exp_datasets.special_tiny_imagenet import *

def get_class_i(x, y, i):
    """
    x: trainset.train_data or testset.test_data
    y: trainset.train_labels or testset.test_labels
    i: class label, a number between 0 to 9
    return: x_i
    """
    # Convert to a numpy array
    y = np.array(y)
    # Locate position of labels that equal to i
    pos_i = np.argwhere(y == i)
    # Convert the result into a 1-D list
    pos_i = list(pos_i[:, 0])
    # Collect all data that match the desired label
    x_i = [x[j] for j in pos_i]

    return x_i


class DatasetMaker(Dataset):
    def __init__(self, datasets, transformFunc):
        """
        datasets: a list of get_class_i outputs, i.e. a list of list of images for selected classes
        """
        self.datasets = datasets
        self.lengths = [len(d) for d in self.datasets]
        self.transformFunc = transformFunc

    def __getitem__(self, i):
        class_label, index_wrt_class = self.index_of_which_bin(self.lengths, i)
        img = self.datasets[class_label][index_wrt_class]
        img = self.transformFunc(img)
        return img, class_label

    def __len__(self):
        return sum(self.lengths)

    def index_of_which_bin(self, bin_sizes, absolute_index, verbose=False):
        """
        Given the absolute index, returns which bin it falls in and which element of that bin it corresponds to.
        """
        # Which class/bin does i fall into?
        accum = np.add.accumulate(bin_sizes)
        if verbose:
            print("accum =", accum)
        bin_index = len(np.argwhere(accum <= absolute_index))
        if verbose:
            print("class_label =", bin_index)
        # Which element of the fallent class/bin does i correspond to?
        index_wrt_class = absolute_index - np.insert(accum, 0, 0)[bin_index]
        if verbose:
            print("index_wrt_class =", index_wrt_class)

        return bin_index, index_wrt_class


def _get_n_fold_datasets_train(x_train, y_train, classDict, transformer, batch_size, class_names=['cat', 'dog']):

    # Let's choose cats (class 3 of CIFAR) and dogs (class 5 of CIFAR) as trainset/testset
    cat_dog_trainset = \
        DatasetMaker(
            [get_class_i(x_train, y_train, classDict[class_names[0]]), get_class_i(x_train, y_train, classDict[class_names[1]])],
            transformer
        )

    kwargs = {'num_workers': 4, 'pin_memory': False}

    # Create datasetLoaders from trainset and testse

    trainsetLoader = DataLoader(cat_dog_trainset, batch_size=batch_size, shuffle=True, **kwargs)
    return trainsetLoader


def _get_n_fold_datasets_test(x_test, y_test, classDict, transformer, batch_size, class_names=['cat', 'dog']):
    # Let's choose cats (class 3 of CIFAR) and dogs (class 5 of CIFAR) as trainset/testset
    cat_dog_testset = \
        DatasetMaker(
            [get_class_i(x_test, y_test, classDict[class_names[0]]),
             get_class_i(x_test, y_test, classDict[class_names[1]])],
            transformer
        )

    kwargs = {'num_workers': 4, 'pin_memory': False}

    # Create datasetLoaders from trainset and testse

    testsetLoader = DataLoader(cat_dog_testset, batch_size=batch_size, shuffle=False, **kwargs)
    return testsetLoader

class Food101Dataset(Dataset):

    def __init__(self, cache_dir, output_size=(512, 512)):
        name = "Food 101 Dataset"
        data_url = 'http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz'
        cache_dir = os.path.join(os.getcwd(), os.path.join(cache_dir, name.replace(' ', '_')))
        super(Food101Dataset, self).__init__(cache_dir, name, data_url, output_size)

    def _load_class_labels(self, metadata_folder):
        with open(os.path.join(metadata_folder, 'classes.txt'), 'r') as labels_file:
            with open(os.path.join(metadata_folder, 'labels.txt'), 'r') as label_name_file:
                for line in zip(labels_file.read().split('\n'), label_name_file.read().split('\n')):
                    if len(line[0]) < 1:
                        continue
                    self.class_labels[line[0]] = line[1]

        for index, label in enumerate(self.class_labels):
            self.class_ids[label] = index

    def _load_json_labels(self, label_list, json_filename, metadata_folder, is_training_data):
        json_file = os.path.join(metadata_folder, json_filename)
        with open(json_file, 'r') as f:
            data = json.loads(f.read())
        for label in self.class_labels:
            for image_id in data[label]:
                image_id = image_id.split('/')[1]
                label_list.append(image_id)
                self.image_labels[image_id] = label
                image_path = os.path.join(self._cache_dir, 'food-101', 'images', label, image_id + '.jpg')
                self.image_paths[image_id] = image_path
                self.is_training_data[image_id] = is_training_data

    def _load_dataset_metadata(self):
        metadata_folder = os.path.join(self._cache_dir, 'food-101', 'meta')
        self._load_class_labels(metadata_folder)
        self._load_json_labels(self.training_ids, 'train.json', metadata_folder, True)
        self._load_json_labels(self.test_ids, 'test.json', metadata_folder, False)


def TinyFood101(batch_size=12, output_size=(32, 32),
                        cache_dir='tmp', selected_classes=list(), no_norm: bool = False, shuffle_test: bool = False):
    # dataset = Food101Dataset(cache_dir)
    # dataset.init(selected_classes=selected_classes)

    RS = transforms.Resize(output_size)
    RC = transforms.RandomCrop(output_size, padding=4)
    RHF = transforms.RandomHorizontalFlip()
    RVF = transforms.RandomVerticalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    TPIL = transforms.ToPILImage()



    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([RS, RC, RHF, TT, NRM]) if not no_norm else transforms.Compose([RS, RHF, TT])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([RS, TT, NRM]) if not no_norm else transforms.Compose([RS, TT])

    train_dataset = torchvision.datasets.ImageFolder(root='./tmp/Food_101_Dataset/food-101/train', transform=transform_with_aug)
    test_dataset = torchvision.datasets.ImageFolder(root='./tmp/Food_101_Dataset/food-101/test', transform=transform_no_aug)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=shuffle_test, num_workers=4)
    train_loader.name = "TinyFood101"
    num_classes = len(selected_classes) if selected_classes else 101
    return train_loader, test_loader, output_size, num_classes



def TinyResizedFood101(batch_size=12, output_size=(256, 256),
                        cache_dir='tmp', selected_classes=list(), no_norm: bool = False, shuffle_test: bool = False):
    # dataset = Food101Dataset(cache_dir)
    # dataset.init(selected_classes=selected_classes)

    RS = transforms.Resize((32, 32))
    RS2 = transforms.Resize(output_size)
    RC = transforms.RandomCrop(output_size, padding=32)
    RHF = transforms.RandomHorizontalFlip()
    RVF = transforms.RandomVerticalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    TPIL = transforms.ToPILImage()



    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([RS, RS2, RC, RHF, TT, NRM]) if not no_norm else transforms.Compose([RS, RHF, TT])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([RS, RS2, TT, NRM]) if not no_norm else transforms.Compose([RS, TT])

    train_dataset = torchvision.datasets.ImageFolder(root='./tmp/Food_101_Dataset/food-101/train', transform=transform_with_aug)
    test_dataset = torchvision.datasets.ImageFolder(root='./tmp/Food_101_Dataset/food-101/test', transform=transform_no_aug)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=shuffle_test, num_workers=4)
    train_loader.name = "TinyResizedFood101"
    num_classes = len(selected_classes) if selected_classes else 101
    return train_loader, test_loader, output_size, num_classes



def Food101(batch_size=12, output_size=(256, 256),
                        cache_dir='tmp', selected_classes=list(), no_norm: bool = False, shuffle_test: bool = False):
    # dataset = Food101Dataset(cache_dir)
    # dataset.init(selected_classes=selected_classes)

    RS = transforms.Resize(output_size)
    RC = transforms.RandomCrop(output_size, padding=32)
    RHF = transforms.RandomHorizontalFlip()
    RVF = transforms.RandomVerticalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    TPIL = transforms.ToPILImage()



    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([RS, RC, RHF, TT, NRM]) if not no_norm else transforms.Compose([RS, RHF, TT])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([RS, TT, NRM]) if not no_norm else transforms.Compose([RS, TT])

    train_dataset = torchvision.datasets.ImageFolder(root='./tmp/Food_101_Dataset/food-101/train', transform=transform_with_aug)
    test_dataset = torchvision.datasets.ImageFolder(root='./tmp/Food_101_Dataset/food-101/test', transform=transform_no_aug)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=shuffle_test, num_workers=4)
    train_loader.name = "Food101"
    num_classes = len(selected_classes) if selected_classes else 101
    return train_loader, test_loader, output_size, num_classes


def CatVsDog(batch_size=12, output_size=(32, 32), cache_dir='tmp'):
    if output_size != (32,32):
        raise RuntimeError("Cifar10 only supports 32x32 images!")

    # Transformations
    RC = transforms.RandomCrop((32, 32), padding=4)
    RHF = transforms.RandomHorizontalFlip()
    RVF = transforms.RandomVerticalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    TPIL = transforms.ToPILImage()

    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([TPIL, RC, RHF, TT, NRM])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([TT, NRM])


    trainset = torchvision.datasets.CIFAR10(root=cache_dir, train=True,
                                            download=True, transform=transform_with_aug)

    testset = torchvision.datasets.CIFAR10(root=cache_dir, train=False,
                                           download=True, transform=transform_no_aug)

    classDict = trainset.class_to_idx

    # Separating trainset/testset data/label
    x_train = trainset.data
    x_test = testset.data
    y_train = trainset.targets
    y_test = testset.targets

    train_loader = _get_n_fold_datasets_train(y_train=y_train,
                                              x_train=x_train,
                                              classDict=classDict,
                                              transformer=transform_with_aug,
                                              batch_size=batch_size,
                                              class_names=['cat', 'dog'])
    test_loader = _get_n_fold_datasets_test(y_test=y_test,
                                            x_test=x_test,
                                            classDict=classDict,
                                            transformer=transform_no_aug,
                                            batch_size=batch_size,
                                            class_names=['cat', 'dog'])

    train_loader.name = "CatVsDog"

    return train_loader, test_loader, (32,32), 2

def SmallTinyImageNet(batch_size=12, output_size=(64, 64), cache_dir='tmp', no_norm: bool = False, test_shuffle=False):
    # Transformations
    # Transformations
    RC = transforms.RandomCrop((64, 64), padding=8)
    RHF = transforms.RandomHorizontalFlip()
    RVF = transforms.RandomVerticalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    TPIL = transforms.ToPILImage()
    RS = transforms.Resize(32)

    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([RC, RHF, RS, TT, NRM]) if not no_norm else transforms.Compose([RC, RHF, TT])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([RS, TT, NRM]) if not no_norm else transforms.Compose([TT])


    trainset = torchvision.datasets.ImageFolder(root='./tmp/tiny-imagenet-200/train/', transform=transform_with_aug)
    testset = torchvision.datasets.ImageFolder(root='./tmp/tiny-imagenet-200/val/ds', transform=transform_no_aug)


    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=3, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=test_shuffle, num_workers=3, pin_memory=False)
    train_loader.name = "SmallTinyImageNet"
    return train_loader, test_loader, (64, 64), 200



def SmallResizedTinyImageNet(batch_size=12, output_size=(64, 64), cache_dir='tmp', no_norm: bool = False, test_shuffle=False):
    # Transformations
    # Transformations
    RC = transforms.RandomCrop((64, 64), padding=8)
    RHF = transforms.RandomHorizontalFlip()
    RVF = transforms.RandomVerticalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    TPIL = transforms.ToPILImage()
    RS = transforms.Resize(32)
    RS2 = transforms.Resize(64)

    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([RC, RHF, RS, RS2, TT, NRM]) if not no_norm else transforms.Compose([RC, RHF, TT])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([RS, RS2, TT, NRM]) if not no_norm else transforms.Compose([TT])


    trainset = torchvision.datasets.ImageFolder(root='./tmp/tiny-imagenet-200/train/', transform=transform_with_aug)
    testset = torchvision.datasets.ImageFolder(root='./tmp/tiny-imagenet-200/val/ds', transform=transform_no_aug)


    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=3, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=test_shuffle, num_workers=3, pin_memory=False)
    train_loader.name = "SmallResizedTinyImageNet"
    return train_loader, test_loader, (64, 64), 200



def TinyImageNet(batch_size=12, output_size=(64, 64), cache_dir='tmp', no_norm: bool = False, test_shuffle=False):
    # Transformations
    # Transformations
    RC = transforms.RandomCrop((64, 64), padding=8)
    RHF = transforms.RandomHorizontalFlip()
    RVF = transforms.RandomVerticalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    TPIL = transforms.ToPILImage()
    RS = transforms.Resize(224)

    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([RC, RHF, TT, NRM]) if not no_norm else transforms.Compose([RC, RHF, TT])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([TT, NRM]) if not no_norm else transforms.Compose([TT])


    trainset = torchvision.datasets.ImageFolder(root='./tmp/tiny-imagenet-200/train/', transform=transform_with_aug)
    testset = torchvision.datasets.ImageFolder(root='./tmp/tiny-imagenet-200/val/ds', transform=transform_no_aug)


    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=3, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=test_shuffle, num_workers=3, pin_memory=False)
    train_loader.name = "TinyImageNet"
    return train_loader, test_loader, (64, 64), 200

def BigTinyImageNet(batch_size=12, output_size=(64, 64), cache_dir='tmp', no_norm: bool = False, test_shuffle=False):
    # Transformations
    # Transformations
    RC = transforms.RandomCrop((64, 64), padding=8)
    RHF = transforms.RandomHorizontalFlip()
    RVF = transforms.RandomVerticalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    TPIL = transforms.ToPILImage()
    RS = transforms.Resize(224)

    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([RC, RHF, RS, TT, NRM])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([RS, TT, NRM])

    trainset = torchvision.datasets.ImageFolder(root='./tmp/tiny-imagenet-200/train/', transform=transform_with_aug)
    testset = torchvision.datasets.ImageFolder(root='./tmp/tiny-imagenet-200/val/ds', transform=transform_no_aug)


    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=3, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=test_shuffle, num_workers=3, pin_memory=False)
    train_loader.name = "BigTinyImageNet"
    return train_loader, test_loader, (64, 64), 200

def g2rgb(x):
    return x.repeat(3, 1, 1)


def BigMNIST(batch_size=12, output_size=(28, 28), cache_dir='tmp'):
    # Transformations
    RC = transforms.RandomCrop((28, 28), padding=3)
    RHF = transforms.RandomHorizontalFlip()
    RVF = transforms.RandomVerticalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    TPIL = transforms.ToPILImage()
    RS = transforms.Resize(224)
    RGB = transforms.Lambda(g2rgb)

    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([RC, RHF, RS, TT, RGB, NRM])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([RS, TT, RGB, NRM])

    trainset = torchvision.datasets.MNIST(root=cache_dir, train=True,
                                            download=True, transform=transform_with_aug)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=3, pin_memory=False)
    testset = torchvision.datasets.MNIST(root=cache_dir, train=False,
                                           download=True, transform=transform_no_aug)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=3, pin_memory=False)
    train_loader.name = "BigMNIST"
    return train_loader, test_loader, (28, 28), 10

def MNIST(batch_size=12, output_size=(28, 28), cache_dir='tmp'):
    # Transformations
    RC = transforms.RandomCrop((28, 28), padding=3)
    RHF = transforms.RandomHorizontalFlip()
    RVF = transforms.RandomVerticalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    TPIL = transforms.ToPILImage()
    RS = transforms.Resize(32)
    RGB = transforms.Lambda(g2rgb)

    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([RC, RHF, RS, TT, RGB, NRM])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([RS, TT, RGB, NRM])


    trainset = torchvision.datasets.MNIST(root=cache_dir, train=True,
                                            download=True, transform=transform_with_aug)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=3, pin_memory=False)
    testset = torchvision.datasets.MNIST(root=cache_dir, train=False,
                                           download=True, transform=transform_no_aug)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=3, pin_memory=False)
    train_loader.name = "MNIST"
    return train_loader, test_loader, (28, 28), 10


def FashionMNIST(batch_size=12, output_size=(28, 28), cache_dir='tmp'):
    # Transformations
    RC = transforms.RandomCrop((28, 28), padding=3)
    RHF = transforms.RandomHorizontalFlip()
    RVF = transforms.RandomVerticalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    TPIL = transforms.ToPILImage()
    RS = transforms.Resize(224)
    RGB = transforms.Lambda(g2rgb)

    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([RC, RHF, TT, RGB, NRM])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([TT, RGB, NRM])


    trainset = torchvision.datasets.FashionMNIST(root=cache_dir, train=True,
                                            download=True, transform=transform_with_aug)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=3, pin_memory=False)
    testset = torchvision.datasets.FashionMNIST(root=cache_dir, train=False,
                                           download=True, transform=transform_no_aug)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=3, pin_memory=False)
    train_loader.name = "FashionMNIST"
    return train_loader, test_loader, (28, 28), 10


def Cifar10(batch_size=12, output_size=(224, 224), cache_dir='tmp'):

    # Transformations
    RC = transforms.RandomCrop((32, 32), padding=4)
    RHF = transforms.RandomHorizontalFlip()
    RVF = transforms.RandomVerticalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    TPIL = transforms.ToPILImage()
    RS = transforms.Resize(224)

    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([RC, RHF, TT, NRM])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([TT, NRM])


    trainset = torchvision.datasets.CIFAR10(root=cache_dir, train=True,
                                            download=True, transform=transform_with_aug)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=8, pin_memory=True)
    testset = torchvision.datasets.CIFAR10(root=cache_dir, train=False,
                                           download=True, transform=transform_no_aug)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=8, pin_memory=True)
    train_loader.name = "Cifar10"
    return train_loader, test_loader, (32,32), 10


def GiantCifar10(batch_size=12, output_size=(224, 224), cache_dir='tmp'):

    # Transformations
    RC = transforms.RandomCrop((32, 32), padding=4)
    RHF = transforms.RandomHorizontalFlip()
    RVF = transforms.RandomVerticalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    TPIL = transforms.ToPILImage()
    RS = transforms.Resize(1024)

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
    train_loader.name = "GiantCifar10"
    return train_loader, test_loader, (1024, 1024), 10

def HugeCifar10(batch_size=12, output_size=(224, 224), cache_dir='tmp'):

    # Transformations
    RC = transforms.RandomCrop((32, 32), padding=4)
    RHF = transforms.RandomHorizontalFlip()
    RVF = transforms.RandomVerticalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    TPIL = transforms.ToPILImage()
    RS = transforms.Resize(448)

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
    train_loader.name = "HugeCifar10"
    return train_loader, test_loader, (448,448), 10


def BigCifar10(batch_size=12, output_size=(224, 224), cache_dir='tmp'):

    # Transformations
    RC = transforms.RandomCrop((32, 32), padding=4)
    RHF = transforms.RandomHorizontalFlip()
    RVF = transforms.RandomVerticalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    TPIL = transforms.ToPILImage()
    RS = transforms.Resize(224)

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
    train_loader.name = "BigCifar10"
    return train_loader, test_loader, (224,224), 10


def MediumCifar10(batch_size=12, output_size=(112, 112), cache_dir='tmp'):

    # Transformations
    RC = transforms.RandomCrop((32, 32), padding=4)
    RHF = transforms.RandomHorizontalFlip()
    RVF = transforms.RandomVerticalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    TPIL = transforms.ToPILImage()
    RS = transforms.Resize(112)

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
    train_loader.name = "MediumCifar10"
    return train_loader, test_loader, (112,112), 10

def SmallCifar10(batch_size=12, output_size=(64, 64), cache_dir='tmp'):

    # Transformations
    RC = transforms.RandomCrop((32, 32), padding=4)
    RHF = transforms.RandomHorizontalFlip()
    RVF = transforms.RandomVerticalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    TPIL = transforms.ToPILImage()
    RS = transforms.Resize(64)

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
    train_loader.name = "SmallCifar10"
    return train_loader, test_loader, (64,64), 10

def Cifar100(batch_size=12, output_size=(32,32), cache_dir='tmp'):


    # Transformations
    RC = transforms.RandomCrop((32, 32), padding=4)
    RHF = transforms.RandomHorizontalFlip()
    RVF = transforms.RandomVerticalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    TPIL = transforms.ToPILImage()
    RS = transforms.Resize(224)

    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([RC, RHF, TT, NRM])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([TT, NRM])

    trainset = torchvision.datasets.CIFAR100(root=cache_dir, train=True,
                                            download=True, transform=transform_with_aug)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=8)
    testset = torchvision.datasets.CIFAR100(root=cache_dir, train=False,
                                           download=True, transform=transform_no_aug)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=8)
    train_loader.name = "Cifar100"
    return train_loader, test_loader, (32, 32), 100


# Lighting data augmentation take from here - https://github.com/eladhoffer/convNet.pytorch/blob/master/preprocess.py
class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()
        return img.add(rgb.view(3, 1, 1).expand_as(img))


def ImageNet(batch_size=12, output_size=(224,224), cache_dir='tmp'):
    size=output_size[0]
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ])
    }


    train_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
        #transforms.RandomHorizontalFlip(),
        #transforms.ColorJitter(.4,.4,.4),
        transforms.ToTensor(),
        #Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_tfms = transforms.Compose([
        transforms.Resize(int(size*1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainset = torchvision.datasets.ImageNet(root="D:\\", transform=train_tfms, split='train')
    testset = torchvision.datasets.ImageNet(root="D:\\", transform=val_tfms, split='val')

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=6, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=6, pin_memory=False)
    train_loader.name = "ImageNet"
    return train_loader, test_loader, (224, 224), 1000

