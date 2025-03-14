# Deep Learning Project: Efficient CIFAR-10 Classification

This repository contains code for training a convolutional neural network (CNN) on the CIFAR-10 dataset using advanced data augmentation techniques, mixed precision training, and a OneCycleLR scheduler. The model architecture is inspired by EfficientNet-style blocks and includes custom residual and squeeze-and-excitation layers. Additionally, the repository provides code for generating predictions on unseen test data and creating a submission CSV file.

## Repository Structure
. ├── README.md # This file ├── dl-project-final-report.tex # LaTeX report for the project ├── train.py # Training script for the CNN model ├── predict.py # Script to generate predictions for test data ├── model.py # Contains model architecture (EfficientNet, MBConv blocks, etc.) ├── data_loader.py # Custom CIFAR-10 data loader with augmentations ├── best_model_retrain.pth # Pretrained model weights (if available) ├── submission_one_last_time_123.csv # Generated submission file (after running predict.py) └── requirements.txt # Python dependencies
