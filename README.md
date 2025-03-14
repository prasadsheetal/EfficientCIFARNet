# Deep Learning Project: Efficient CIFAR-10 Classification

This repository contains code for training a convolutional neural network (CNN) on the CIFAR-10 dataset using advanced data augmentation techniques, mixed precision training, and a OneCycleLR scheduler. The model architecture is inspired by EfficientNet-style blocks and includes custom residual and squeeze-and-excitation layers. Additionally, the repository provides code for generating predictions on unseen test data and creating a submission CSV file.

## Repository Structure

. ├── README.md # This file ├── dl-project-final-report.tex # LaTeX report for the project ├── train.py # Training script for the CNN model ├── predict.py # Script to generate predictions for test data ├── model.py # Contains model architecture (EfficientNet, MBConv blocks, etc.) ├── data_loader.py # Custom CIFAR-10 data loader with augmentations ├── best_model_retrain.pth # Pretrained model weights (if available) ├── submission_one_last_time_123.csv # Generated submission file (after running predict.py) └── requirements.txt # Python dependencies

markdown
Copy

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
Installation
Clone this repository:

bash
Copy
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
Install the dependencies:

bash
Copy
pip install -r requirements.txt
Usage
Training the Model
To train the model on CIFAR-10 with advanced data augmentations and mixed precision training, run:

bash
Copy
python train.py
This script performs the following:

Loads and preprocesses CIFAR-10 data using a custom CIFAR class.
Builds the EfficientNet-style model.
Trains the model for a specified number of epochs (e.g., 150 or 300) using OneCycleLR scheduling.
Saves the best performing model weights to best_model_retrain.pth.
Generating Predictions
Once training is complete and the best model is saved, you can generate predictions on unseen test data by running:

bash
Copy
python predict.py
This script:

Loads the test data from a pickle file.
Preprocesses the data using the UnseenDataset class.
Uses the trained model to predict labels for the test set.
Saves the predictions in a CSV file named submission_one_last_time_123.csv.
Model Architecture
The model is based on an EfficientNet-style architecture with the following components:

Stem: Initial convolution and squeeze-excitation block.
Block Stack: A series of MBConv blocks with residual connections and depthwise convolutions.
Head: A fully connected layer with dropout, followed by a softmax output.
A TikZ diagram of the architecture is provided in the LaTeX report (dl-project-final-report.tex).

Evaluation
The training process monitors both training and validation loss/accuracy. Key metrics include:

Training Accuracy: ~98%
Validation Accuracy: ~93%
Test Accuracy: ~92.7%
A parameter limit of 5,000,000 trainable parameters is enforced, ensuring the model remains efficient.

Code Overview
Model Initialization:
The model is instantiated, initialized with Kaiming Normal, and loaded with pretrained weights if available.

Training Loop:
The training loop uses PyTorch Ignite for supervised training, mixed precision for efficiency, and a OneCycleLR scheduler for learning rate adjustment.

Prediction Pipeline:
The prediction script loads test data, preprocesses it, runs inference with the trained model, and writes the output to a CSV file.

Contributing
Contributions to enhance the model, improve data preprocessing, or refine training techniques are welcome. Please open an issue or submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Contact
For any questions or suggestions, please contact:

Sheetal Prasad (sp7990@nyu.edu)
Naveenraj Kamalakannan (nk3940@nyu.edu)
Harish Balaji (hb2917@nyu.edu)
pgsql
Copy

Save the file and commit it to your Git repository. This README provides an overview of the project, its structure, requirements, and usage instructions. Feel free to adjust any sections to suit your project's specific details.






