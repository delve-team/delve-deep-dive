from datasets import ImageNet, TinyImageNet, Cifar10, Cifar100, Food101
from models import TinyCAE, TinyCAEPCA, BIGCAEPCA
import torch
from torch import nn
from pca_layers import change_all_pca_layer_thresholds
from os.path import join, exists
from os import mkdir
from skimage.io import imshow, show
import numpy as np
from matplotlib.pyplot import title, savefig

if __name__ == '__main__':
    state_path = './logs/TinyCAE/Food101/TinyCAE_bs128_e20_idcentered3.pt'
    device = 'cpu'
    #torch.cuda.set_device(device)



    cae = TinyCAE().to(device)
    thresh = 10.0
    _, test, _, _ = Food101(1, no_norm=True, shuffle_test=True)
    path = './CenteredDenseFood101Samples2'
    if not exists(path):
        print('created dir')
        mkdir(path=path)

    cae.load_state_dict(torch.load(state_path)['model_state_dict'])

    cae.eval()
    print(cae.pca_layer.centering)
    change_all_pca_layer_thresholds(thresh, cae, verbose=True)
    cae.to(device)
    for i,(data, _) in enumerate(test):
        data = data.to(device)
        for thresh in [10.0, 0.9999, 0.999, 0.995, 0.99, 0.95, 0.9, 0.5]:
            sat, in_dim, fs_dim = change_all_pca_layer_thresholds(thresh, cae)
            d = data.squeeze().cpu().numpy().swapaxes(0, -1).swapaxes(0, 1)
            pred = cae(data)
            p = pred.squeeze().cpu().detach().numpy().swapaxes(0, -1).swapaxes(0, 1)
            im = np.append(d, p, axis=1)
            #im = p
            imshow(im)
            title(f'Reconstruction with PCA-Threshold {round(thresh*100, 2) if thresh != 10 else 100}%, AvgSat: {round(np.mean(sat),2) if thresh != 10 else 100}%')
            savefig(join(path, f'{i}sample-thresh{round(thresh*10000) if thresh != 10 else 10000}-{cae.pca_layer.reduced_transformation_matrix.shape[-1]}f-{8192}c-{round(cae.pca_layer.reduced_transformation_matrix.shape[-1] / 8192*100, 2)}s.jpg'))
