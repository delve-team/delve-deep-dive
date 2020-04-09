import torchvision.datasets.imagenet as imnet
from os.path import join, curdir

train_folder = join(curdir,'tmp','ImgNet','train')
valid_folder = join(curdir,'tmp','ImgNet','valid')
root = 'D:\\'

#imnet.parse_train_archive(root)
imnet.parse_devkit_archive(root)
imnet.parse_val_archive(root)