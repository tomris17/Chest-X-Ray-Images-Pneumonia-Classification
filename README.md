Chest X-Ray Pneumonia Detection using CNN

An end-to-end Deep Learning project designed to automatically detect Pneumonia from pediatric chest X-ray images using a Convolutional Neural Network (CNN) architecture, deployed with an interactive Streamlit application.

Project Overview

This project aims to leverage Artificial Intelligence for automated medical image analysis. By processing chest radiography scans, the developed CNN model distinguishes between healthy controls (NORMAL) and positive pneumonia cases (PNEUMONIA), helping streamline preliminary medical assessments with high diagnostic reliability.

Dataset: Kaggle - Chest X-Ray Images (Pneumonia)
Model Architecture: Custom Convolutional Neural Network (CNN)
Primary Metric: Test Accuracy (98.8%)
Deployment: Streamlit Web Application
Dataset Structure

The dataset contains pre-split X-ray images organized as follows:

chest_xray/
│
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
└── test/
    ├── NORMAL/
    └── PNEUMONIA/

Image Preprocessing

Image Reading: OpenCV (cv2) is used for reading X-ray images.
Color Conversion: Images are converted to RGB format.
Resizing: Target dimensions are set to 128x128.
Normalization: Pixel values are normalized using $X / 255.0$.
Data Splitting: train_test_split is used with label stratification.
Model Architecture

The CNN architecture consists of:

Feature Extraction: Stacked Conv2D and MaxPooling2D layers.
Flattening: A Flatten layer transitions the extracted features into a fully connected network.
Fully Connected Layer: Dense layers are used for classification.
Regularization: Dropout with a rate of 0.5 is applied to prevent overfitting.
Output Layer: A Sigmoid activation function is used for binary classification.
Training & Evaluation
Optimization: Adam
Loss Function: binary_crossentropy
Class Imbalance: Balanced class_weight adjustments are used during training.
Performance & Results
Test Accuracy: 98.8%
Evaluation Metrics: Accuracy and Loss curves are plotted across epochs.
Confusion Matrix: A detailed Confusion Matrix is used to evaluate classification performance.

Streamlit Web App Deployment

An interactive frontend built with Streamlit allows users to upload custom X-ray scans and receive instant predictions with confidence scores.

Running the App Locally
1. Clone the Repository
git clone https://github.com/your-username/chest-xray-pneumonia-detection.git
cd chest-xray-pneumonia-detection
