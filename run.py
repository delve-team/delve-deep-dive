import argparse
import models
import datasets
import sys
import types

from trainer import Trainer

parser = argparse.ArgumentParser(description='Train a network on a dataset')
parser.add_argument('-n', '--network', dest='model_name', action='store')
parser.add_argument('-d', '--dataset', dest='dataset_name', action='store')
parser.add_argument('-b', '-batch-size', dest='batch_size', action='store')

def parse_model(model_name, shape, num_classes):
    try:
        model = models.__dict__[model_name](input_size=shape, num_classes=num_classes)
    except KeyError:
        raise NameError("%s doesn't exist." % model_name)
    return model


def parse_dataset(dataset_name, batch_size):
    batch_size = int(batch_size)
    try:
        train_loader, test_loader, shape, num_classes = datasets.__dict__[dataset_name](batch_size=batch_size)
    except KeyError:
        raise NameError("%s doesn't exist." % dataset_name)
    return train_loader, test_loader, shape, num_classes




if __name__ == '__main__':
    args = parser.parse_args()
    train_loader, test_loader, shape, num_classes = parse_dataset(args.dataset_name, args.batch_size)
    model = parse_model(args.model_name, shape, num_classes)
    trainer = Trainer(model, train_loader, test_loader)
    trainer.train()
