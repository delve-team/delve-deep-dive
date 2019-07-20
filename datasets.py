import os
from imageio import imread
import tarfile
from urllib.request import urlretrieve
from collections import OrderedDict

from dataclasses import dataclass, field
import json
import scipy
import scipy.misc
import numpy as np

import torch.utils.data


@dataclass
class Dataset(torch.utils.data.Dataset):
    """ The Dataset class downloads and exposes an image dataset.

    A dataset is structured around image IDs. These identify each unique picture.
    Dictionaries then allow the user to lookup properties of each image,
    such as its class label or whether it is in the training or test dataset.
    The dataset class will preprocess the data and make sure each image has
    the required output dimensions.

    To be compatible with Torch, each image also has an index, ranging from 0 to
    len(dataset) - 1. The indices are derived from the image's position in the images
    map. The Dataset class exposes the images by index through the __getitem__ function.

    The dataset also exposes the training indices and test indices. These can then be fed
    into a SubsetRandomSampler to generate a dataloader that fetches data from the
    training set and test set respectively.

    To implement a dataset, subclass the Dataset function and implement the
    _load_dataset_metadata function and set the _data_url variable.
    The dataset will then download, unpack and initialise the data.

    When loading the data, the user can optionally specify a list of class labels.
    The dataset will then remove any images that do not have any of the given
    class labels. This will be done before loading the data.
    The images are guaranteed to be removed from the training set and test
    set list, as well as the images dict and the _images_iter dict (and by implication
    the __getitem__ interface). The dataset length will also be adjusted to correspond
    to the number of images actually loaded.

    Attributes:
        _cache_dir          The location of the dataset on disk.
        name                The name of the dataset.
        _data_url           The URL specifying the location of the dataset.
        image_paths         A map from image IDs to paths on disk.
        images              A map from image IDs to the initialised data.
        image_labels        A map from image IDs to the image's class label.
        class_labels        A map from the class labels to the class names.
        is_training_data    A map from the image IDs to a bool specifying if the image is in the training set.
        training_ids        A list of the IDs of training images.
        test_ids            A list of the IDs of test images.
        _images_iter        A map from the image's index to the image data.
    """

    _cache_dir: str
    name: str
    _data_url: str
    output_size: (int, int)
    image_paths: dict = field(default_factory=OrderedDict)
    images: dict = field(default_factory=OrderedDict)
    image_labels: dict = field(default_factory=dict)
    class_labels: dict = field(default_factory=dict)
    is_training_data: dict = field(default_factory=dict)
    training_ids: list = field(default_factory=list)
    test_ids: list = field(default_factory=list)
    _images_iter: dict = field(default_factory=OrderedDict)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self._images_iter[index]

    def training_indices(self):
        return [index for (index, key) in enumerate(self.images.keys()) if self.is_training_data[key]]

    def test_indices(self):
        return [index for (index, key) in enumerate(self.images.keys()) if not self.is_training_data[key]]

    def _setup_cache_dir(self):
        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir)

    def _fetch_and_unpack_tar(self):
        tarball = os.path.join(self._cache_dir, "dataset.tgz")
        if not (os.path.exists(tarball)):
            print("Fetching data from {}".format(self._data_url))
            filename, headers = urlretrieve(self._data_url, tarball)
            tar = tarfile.open(tarball)
            tar.extractall(path=self._cache_dir)
            tar.close()

    def _fetch_data(self):
        # For now, all data is in tar files
        self._setup_cache_dir()
        self._fetch_and_unpack_tar()

    def _remove_unwanted_classes(self, selected_classes):
        if len(selected_classes) == 0:
            return
        self.training_ids = [image_id for image_id in self.training_ids \
                             if self.image_labels[image_id] in selected_classes]
        self.test_ids = [image_id for image_id in self.test_ids \
                         if self.image_labels[image_id] in selected_classes]
        for image_id in list(self.image_paths.keys()):
            if self.image_labels[image_id] not in selected_classes:
                del self.image_paths[image_id]
                del self.image_labels[image_id]


    def _init_data(self):
        for index, (image_id, image_path) in enumerate(self.image_paths.items()):
            image = imread(image_path)
            image_interp = scipy.misc.imresize(image, (*self.output_size, 3))
            image_interp = np.swapaxes(image_interp, 0, 2)
            image = torch.from_numpy(image_interp / 255)
            self.images[image_id] = image
            self._images_iter[index] = image


    def init(self, selected_classes: list = list()):
        self._fetch_data()
        self._load_dataset_metadata()
        self._remove_unwanted_classes(selected_classes=selected_classes)
        self._init_data()


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
        self._load_json_labels(self.test_ids, 'test.json', metadata_folder, True)
