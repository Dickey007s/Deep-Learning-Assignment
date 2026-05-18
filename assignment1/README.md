# Deep Learning Assignment Repository

This is my personal repository for Deep Learning course assignments.

## Contents

### Assignment 1 - Image Classification

- **k-Nearest Neighbor (kNN)**: Implemented kNN classifier with L2 distance computation
  - `knn.ipynb`: Main notebook with experiments and cross-validation
  - `cs231n/classifiers/k_nearest_neighbor.py`: kNN classifier implementation
    - Double loop distance computation
    - Single loop distance computation (vectorized over training data)
    - Fully vectorized distance computation (no loops)

- **Other notebooks** (to be completed):
  - `softmax.ipynb`: Softmax classifier
  - `two_layer_net.ipynb`: Two-layer neural network
  - `features.ipynb`: Feature extraction
  - `FullyConnectedNets.ipynb`: Fully connected neural networks

## Dataset

The CIFAR-10 dataset is used for image classification tasks. It contains:
- 50,000 training images
- 10,000 test images
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Note: Dataset files are not included in this repository due to size constraints. Run `get_datasets.sh` in `cs231n/datasets/` to download the data.

## Course Information

This repository contains assignments from the CS231n: Convolutional Neural Networks for Visual Recognition course.