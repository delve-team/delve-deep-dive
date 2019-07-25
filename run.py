import argparse
import models
import datasets
import json
import sys
import types

from trainer import Trainer

parser = argparse.ArgumentParser(description='Train a network on a dataset')
parser.add_argument('-n', '--network', dest='model_name', action='store', default='vgg11')
parser.add_argument('-d', '--dataset', dest='dataset_name', action='store', default='Cifar10')
parser.add_argument('-b', '--batch-size', dest='batch_size', action='store', default=32)
parser.add_argument('-o', '--output', dest='output', action='store', default='logs')
parser.add_argument('-c', '--compute-device', dest='device', action='store', default='cpu')
parser.add_argument('-r', '--run_id', dest='run_id', action='store', default=0)
parser.add_argument('-cf', '--config', dest='json_file', action='store', default=None)


def parse_model(model_name, shape, num_classes):
    try:
        model = models.__dict__[model_name](input_size=shape, num_classes=num_classes)
        print(model)
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
    if args.json_file is None:
        print('Starting manual run')
        train_loader, test_loader, shape, num_classes = parse_dataset(args.dataset_name, args.batch_size)
        model = parse_model(args.model_name, shape, num_classes)
        trainer = Trainer(model, train_loader, test_loader, logs_dir=args.output, device=args.device, run_id=args.run_id)
        trainer.train()
    else:
        print('Automatized experiment schedule enabled using', args.json_file)
        config_dict = json.load(open(args.json_file, 'r'))

        run_num = 0
        for batch_size in config_dict['batch_sizes']:
            for model in config_dict['models']:
                run_num += 1
                print('Runnin Experiment', run_num, 'of', len(config_dict['batch_sizes'])*len(config_dict['models']))
                train_loader, test_loader, shape, num_classes = parse_dataset(config_dict['dataset'], batch_size)
                model = parse_model(model, shape, num_classes)
                trainer = Trainer(model, train_loader, test_loader, logs_dir=args.output, device=args.device, run_id=args.run_id)
                trainer.train()
