import argparse
import models
import datasets
import json
import torch
import sys
import types
from pca_layers import change_all_pca_layer_thresholds
from pca_layers import change_all_pca_layer_centering
from thop import profile

from trainer import Trainer

parser = argparse.ArgumentParser(description='Train a network on a dataset')
parser.add_argument('-n', '--network', dest='model_name', action='store', default='vgg11')
parser.add_argument('-d', '--dataset', dest='dataset_name', action='store', default='Cifar10')
parser.add_argument('-b', '--batch-size', dest='batch_size', action='store', default=32)
parser.add_argument('-o', '--output', dest='output', action='store', default='logs')
parser.add_argument('-c', '--compute-device', dest='device', action='store', default='cpu')
parser.add_argument('-r', '--run_id', dest='run_id', action='store', default=0)
parser.add_argument('-cf', '--config', dest='json_file', action='store', default=None)
parser.add_argument('-cs', '--saturation-device', dest='sat_device', type=str, default=None, action='store')


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
    if args.json_file is None:
        print('Starting manual run')
        train_loader, test_loader, shape, num_classes = parse_dataset(args.dataset_name, args.batch_size)
        model = parse_model(args.model_name, shape, num_classes)
        trainer = Trainer(model, train_loader, test_loader, logs_dir=args.output, device=args.device, run_id=args.run_id)
        trainer.train()
    else:
        print('Automatized experiment schedule enabled using', args.json_file)
        config_dict = json.load(open(args.json_file, 'r'))
        thresholds = [.99] if not 'threshs' in config_dict else config_dict['threshs']
        dss = config_dict['dataset'] if isinstance(config_dict['dataset'], list) else [config_dict['dataset']]
        optimizer = config_dict['optimizer']
        run_num = 0
        print(thresholds)
        for dataset in dss:
            for thresh in thresholds:
                for batch_size in config_dict['batch_sizes']:
                    for model in config_dict['models']:
                        run_num += 1
                        print('Running Experiment', run_num, 'of', len(config_dict['batch_sizes'])*len(config_dict['models']*len(thresholds))*len(dss))
                        train_loader, test_loader, shape, num_classes = parse_dataset(dataset, batch_size)
                        model = parse_model(model, shape, num_classes)
                        change_all_pca_layer_thresholds(thresh, model, verbose=True)
                        if 'centering' in config_dict:
                            change_all_pca_layer_centering(centering=config_dict['centering'], network=model, verbose=True)
                        else:
                            change_all_pca_layer_centering(centering=False, network=model, verbose=True)
                        conv_method = 'channelwise' if 'conv_method' not in config_dict else config_dict['conv_method']
                        trainer = Trainer(model,
                                          train_loader,
                                          test_loader,
                                          logs_dir=args.output,
                                          device=args.device,
                                          run_id=args.run_id,
                                          epochs=config_dict['epochs'],
                                          batch_size=batch_size,
                                          optimizer=optimizer,
                                          plot=True,
                                          compute_top_k=True if dataset == 'ImageNet' else False,
                                          data_prallel=False if torch.cuda.device_count() > 1 and dataset == 'ImageNet' else False,
                                          saturation_device=args.sat_device,
                                          conv_method=conv_method,
                                          thresh=thresh)
                        #input = torch.randn(1, 3, 32, 32)
                        #flops, params = profile(model, inputs=(input,))
                        #print(flops/1000000, 'MFLOP')
                        trainer.train()
