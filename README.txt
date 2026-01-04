CENG 476 – Deep Learning Project
Real vs Fake Face Classification

This project focuses on binary classification of facial images as Real or Fake
using deep learning techniques.

Project Structure

Files

train_simplecnn.py
Baseline CNN trained from scratch and used as a reference model.

train_resnet50.py
Transfer learning approach using a pre-trained ResNet50 model with frozen backbone.

train_resnet50_finetune.py
Final and best-performing model. ResNet50 with fine-tuning applied to layer4
and the classifier.

Outputs

Each training script saves:

Training and validation loss and accuracy curves

Best model checkpoint

Test accuracy

Classification report

Confusion matrix

Dataset

The dataset is a binary face image dataset consisting of two classes: real and fake.

The dataset root directory is specified using the CFG.base_dir variable inside
each training script.

Example configuration:

CFG.base_dir = "/path/to/dataset"

Expected directory structure:

dataset/
 ├── train/
 │    ├── real/
 │    └── fake/
 ├── val/
 │    ├── real/
 │    └── fake/
 └── test/
      ├── real/
      └── fake/


Installation

Install required dependencies using:

pip install -r requirements.txt

How to Run

Run the training scripts in order:

python train_simplecnn.py
python train_resnet50.py
python train_resnet50_finetune.py

Make sure CFG.base_dir is set correctly before running the scripts.