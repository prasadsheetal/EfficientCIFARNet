# Deep Learning Project: Efficient CIFAR-10 Classification

This repository contains code for training a convolutional neural network (CNN) on the CIFAR-10 dataset using advanced data augmentation techniques, mixed precision training, and a OneCycleLR scheduler. The model architecture is inspired by EfficientNet-style blocks and includes custom residual and squeeze-and-excitation layers.

## Repository Structure
.
├── README.md
├── dl-project-final-report.tex
├── train.py
├── predict.py
├── model.py
├── data_loader.py
├── best_model_retrain.pth
├── submission_one_last_time_123.csv
└── requirements.txt

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
- (Optional) torchsummary

Install dependencies:
```bash
pip install -r requirements.txt
Installation
bash
Copy
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
Usage
Training
bash
Copy
python train.py
Implements OneCycleLR scheduling

Uses mixed precision training

Saves best weights to best_model_retrain.pth

Prediction
bash
Copy
python predict.py
Generates submission_one_last_time_123.csv

Requires test data in pickle format

Model Architecture
EfficientNet-style components:

Stem: Conv + Squeeze-Excitation

MBConv blocks with depthwise convolutions

Head: FC layer with dropout

Diagram available in dl-project-final-report.tex

Performance
Metric	Value
Training Acc	~98%
Validation Acc	~93%
Test Acc	~92.7%
Code Structure
model.py: EfficientNet implementation

data_loader.py: Custom transforms/augmentations

train.py: Training pipeline with metrics tracking

predict.py: Inference pipeline for test data

Contributing
Contributions welcome! Open issues/PRs for:

Model architecture improvements

Enhanced data augmentations

Training optimization

License
MIT License

Contact
Sheetal Prasad (sp7990@nyu.edu)

Naveenraj Kamalakannan (nk3940@nyu.edu)

Harish Balaji (hb2917@nyu.edu)
