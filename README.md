# CLRNet

CLRNet is a network framework for hierarchically constructing potential energy surfaces, integrating machine learning methods including graph neural networks, clustering, and local regression.

## Install environments

The following dependent  packages are required：

ase = 3.22.1 (Single molecule model building)

schnetpack = 2.0.4 (Database construction)

torch = 2.2.2 (Machine learning method implementation)

pytorch_lightning = 2.2.1 (Machine learning architecture implementation)

scikit-learn = 1.5.2 (Clustering method implementation)

umap-learning = 0.5.7 (Dimensionality reduction method implementation)

## Brief introduction

Clrnet_main: This script sets up and trains a machine learning model with data preparation, model definition, and training configuration included.

feature_extraction: This code defines a Cfconv neural network module for processing atomic data.

energy regression : This code defines the Clrnet class for processesing molecular features to predict PES, and allows for both training and prediction modes with optional smoothing of cluster contributions.

task_setting : The Task class is a PyTorch Lightning module that manages the training, validation, and testing of a model for energy prediction, incorporating loss computation, metrics tracking, and optimizer configuration, while also handling model state saving and epoch-end cleanup.

## Usage

Default running:

```
python train.py
```

Specify configuration file:

```
python train.py --config custom_config.yaml
```

Override parameters:

```
python train.py --batch_size 32 --learning_rate 1e-4 --max_epochs 200
```
