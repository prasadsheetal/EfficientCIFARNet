# Deep Learning Project: Efficient CIFAR-10 Classification

This repository contains code for training a convolutional neural network (CNN) on the CIFAR-10 dataset using advanced data augmentation techniques, mixed precision training, and a OneCycleLR scheduler. The model architecture is inspired by EfficientNet-style blocks and includes custom residual and squeeze-and-excitation layers. Additionally, the repository provides code for generating predictions on unseen test data and creating a submission CSV file.

## Repository Structure

## Requirements

- Python 3.7+
- PyTorch 1.7+
- TorchVision
- Ignite
- NumPy
- Pandas
- Matplotlib
- tqdm
- (Optional) torchsummary for model summary

You can install the required packages using:

```bash
pip install -r requirements.txt
