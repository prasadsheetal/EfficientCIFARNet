# Deep Learning Project: Efficient CIFAR-10 Classification

This repository contains code for training a convolutional neural network (CNN) on the CIFAR-10 dataset using advanced data augmentation techniques, mixed precision training, and a OneCycleLR scheduler. The model architecture is inspired by EfficientNet-style blocks and includes custom residual and squeeze-and-excitation layers.

## Repository Structure

- README.md  
- dl-project-final-report.tex  
- train.py  
- predict.py  
- model.py  
- data_loader.py  
- best_model_retrain.pth  
- submission_one_last_time_123.csv  
- requirements.txt

## Requirements
- Python 3.7+
- PyTorch 1.7+
- TorchVision
- Ignite
- NumPy
- Pandas
- Matplotlib
- tqdm
- (Optional) torchsummary

## Installation

```bash
pip install -r requirements.txt
```
## Usage

### Installation

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```

## Training

```bash
python train.py
```
This script performs the following:

- Loads and preprocesses CIFAR-10 data using a custom `CIFAR` class.
- Builds the EfficientNet-style model.
- Trains the model for a specified number of epochs (e.g., 150 or 300) using OneCycleLR scheduling.
- Saves the best performing model weights to `best_model_retrain.pth`.

## Prediction

```bash
python predict.py
```
This script:

- Loads the test data from a pickle file.
- Preprocesses the data using the `UnseenDataset` class.
- Uses the trained model to predict labels for the test set.
- Saves the predictions in a CSV file named `submission_one_last_time_123.csv`.

## Model Architecture

The model is based on an EfficientNet-style architecture with the following components:

- **Stem:** Initial convolution and squeeze-excitation block.
- **Block Stack:** A series of MBConv blocks with residual connections and depthwise convolutions.
- **Head:** A fully connected layer with dropout, followed by a softmax output.

A TikZ diagram of the architecture is provided in the LaTeX report (`dl-project-final-report.tex`).

## Evaluation

The training process monitors both training and validation loss/accuracy. Key metrics include:
- **Training Accuracy:** ~98%
- **Validation Accuracy:** ~93%
- **Test Accuracy:** ~92.7%

A parameter limit of 5,000,000 trainable parameters is enforced, ensuring the model remains efficient.

## Code Overview

- **Model Initialization:**  
  The model is instantiated, initialized with Kaiming Normal, and loaded with pretrained weights if available.

- **Training Loop:**  
  The training loop uses PyTorch Ignite for supervised training, mixed precision for efficiency, and a OneCycleLR scheduler for learning rate adjustment.

- **Prediction Pipeline:**  
  The prediction script loads test data, preprocesses it, runs inference with the trained model, and writes the output to a CSV file.

## Contact

For any questions or suggestions, please contact:

- **Sheetal Prasad** (sp7990@nyu.edu)
- **Naveenraj Kamalakannan** (nk3940@nyu.edu)
- **Harish Balaji** (hb2917@nyu.edu)

