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

## Usage

Clrnet_main: This script sets up and trains a machine learning model with data preparation, model definition, and training configuration included.

feature_extraction: This code defines a Cfconv neural network module for processing atomic data.

energy regression :

task_setting :
