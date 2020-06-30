# This is the repository used for conducting all experiments in the submitted work


# Important files and folder

## important files

	run.py				This file is used for training models. The training is configured by a json file.
					The GPU to use and other things can be modified additionally using command-line arguments

	classifier_inferencer.py	Uses the same config as run.py for configuration. This file will run inferences using different
					delta thresholds.

	probe_data_collecter.py		Used for collecting the data from the probes, this uses also the same config and run.py and classifier_inferencer.py
					This file will create a folder with the different layer outputs

	train_probes.py			When pointed to the folder created by probe_data_collecter.py this script will train logistic regressions on all extracted layers.

	

## configs

{
  "models" : ["a_model", "another_model"],		// the models are obtained over reflection. The models are implemented in models.py any implemented model in this file can be used
  "dataset": ["Dataset1", "Dataset2", "Dataset3"],	// references the dataset, works similar to referencing the models. Datasets are implemented in datasets.py
  "epochs":   42,					// number of epochs
  "batch_sizes": [32],					// The list of batch sizes to test
  "optimizer": "adam",					// The optimizer used for training
  "conv_method": "channelwise",				// pooling strategy for saturation. Channelwise is the default, which is also used to generate all result in the paper
  "threshs": [10.0],					// deprocated parameter, generally only used for repeating a specific run. This number must be larger than 10
  "centering": true,					// subtractss the mean from the covariance matrix. True by default, only briefly used for testing how zero-centered the data actually is
  "downsampling": [32]					// downsampling strategy. The feature map for computing saturation is downsamples the height and width described by this number. May boost performance.
}

All parameters that are represented as list form a carthesian product. All resulting tuples resulting from this product are experiments that will be conducted sequentially.

## Enabling and Disabling PCA Layers
In the models.py file is local variable PCA, if this variable is set to True PCA Layers are enabled for some models. Please check the implementation.

## Downloading Datasets
Automatic downloads are enabled for MNIST, CIFAR10, CIFAR100 and CatVsDog. Food101, TinyImageNet and ImageNet need to be downloaded seperatly. There are multiple scripts for preparing the datasets correctly in this repository.