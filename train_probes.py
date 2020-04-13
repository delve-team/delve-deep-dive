import os
import numpy as np
import sys
from typing import List, Tuple, Union, Dict
from itertools import product
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression as LogisticRegressionModel
import argparse
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        outputs = torch.nn.functional.softmax(outputs)
        return outputs


def dataset_from_array(data: np.ndarray, targets: np.ndarray):
    tensor_x = torch.Tensor(data)  # transform to torch tensor
    tensor_y = torch.Tensor(targets).long()
    dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    dataloader = DataLoader(dataset, batch_size=1000)  # create your dataloader
    return dataloader


def train(model, train_loader, criterion=torch.nn.CrossEntropyLoss(), epochs=100, device='cuda:0'):
    iter = 0
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), 0.01)
    loss = 0
    for epoch in range(int(epochs)):
        correct = 0
        total = 0
        for i, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            iter += 1
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # for gpu, bring the predicted and labels back to cpu fro python operations to work
            correct += (predicted == labels).sum()
        accuracy = round(100 * correct.cpu().numpy() / total, 2)
        print("Epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), accuracy))
    return model


def filter_files_by_string_key(files: List[str], key: str) -> List[str]:
    return [file for file in files if key in file]


def seperate_labels_from_data(files: List[str]) -> Tuple[List[str], List[str]]:
    data_files = [file for file in files if '-labels' not in file]
    label_file = [file for file in files if '-labels' in file]
    return data_files, label_file


def get_all_npy_files(folder: str) -> List[str]:
    all_files = os.listdir(folder)
    filtered_files = filter_files_by_string_key(all_files, '.p')
    full_paths = [os.path.join(folder, file) for file in filtered_files]
    return full_paths


def obtain_all_dataset(folder: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    all_files = get_all_npy_files(folder)
    data, labels = seperate_labels_from_data(all_files)
    train_data, train_label = filter_files_by_string_key(data, 'train-'), filter_files_by_string_key(labels, 'train-')
    eval_data, eval_label = filter_files_by_string_key(data, 'eval-'), filter_files_by_string_key(labels, 'eval-')
    train_set = [elem for elem in product(train_data, train_label)]
    eval_set = [elem for elem in product(eval_data, eval_label)]
    return train_set, eval_set


def loadall(filename: str) -> np.ndarray:
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

def load(filename: str) -> np.ndarray:
    return np.vstack([batch for batch in loadall(filename)])


def get_data_annd_labels(data_path: str, label_path: str) -> Tuple[np.ndarray, np.ndarray]:
    return load(data_path), np.squeeze(load(label_path))


def train_model(data_path: str, labels_path: str) -> LogisticRegression:
    print('Loading training data from', data_path)
    data, labels = get_data_annd_labels(data_path, labels_path)
    #data_loader = dataset_from_array(data, labels)
    print('Training data obtained with shape', data.shape)
    #model = LogisticRegression(input_dim=data.shape[-1], output_dim=len(np.unique(labels)))
    model = LogisticRegressionModel(multi_class='multinomial', n_jobs=6, solver='saga', verbose=1).fit(data, labels)#train(model, data_loader)
    return model


def obtain_accuracy(model: LogisticRegression, data_path, label_path: str) -> float:
    data, labels = get_data_annd_labels(data_path, label_path)
    #model = model.cpu()
    #t_data = torch.from_numpy(data)
    #out = model(t_data)
    #_, preds = torch.max(out.data, 1)
    #preds = preds.cpu().numpy()
    preds = model.predict(data)

    return accuracy_score(labels, preds)


def train_model_for_data(train_set: Tuple[str, str], eval_set: Tuple[str, str]):
    print('Training model')
    model = train_model(*train_set)
    print('Obtaining metrics')
    train_acc = obtain_accuracy(model, *train_set)
    eval_acc = obtain_accuracy(model, *eval_set)
    print(os.path.basename(train_set[0]))
    print('Train acc', train_acc)
    print('Eval acc:', eval_acc)
    return train_acc, eval_acc

import pandas as pd
if __name__ == '__main__':
    names, t_accs, e_accs = [], [], []
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='folder', type=str, default=None, help='data folder')
    args = parser.parse_args()
    train_set, eval_set = obtain_all_dataset(args.folder)
    print(len(train_set), len(eval_set))
    train_set.reverse(), eval_set.reverse()
    for train_data, eval_data in zip(train_set, eval_set):
        train_acc, eval_acc = train_model_for_data(train_data, eval_data)
        names.append(os.path.basename(train_data[0][:-2]))
        t_accs.append(train_acc)
        e_accs.append(eval_acc)
        pd.DataFrame.from_dict(
            {
                'name': names,
                'train_acc': t_accs,
                'eval_acc': eval_acc
            }
        ).to_csv('results_vgg16.csv', sep=';')
