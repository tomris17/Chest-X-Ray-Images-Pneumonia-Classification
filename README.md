# Chest X-Ray Pneumonia Detection

An end-to-end Deep Learning project to automatically detect Pneumonia from chest X-ray images using Custom CNN and **MobileNetV2 Transfer Learning**, deployed via an interactive Streamlit web application.

---

## Project Overview

This project builds an automated medical diagnostic tool to distinguish between `NORMAL` and `PNEUMONIA` X-ray scans. By incorporating **MobileNetV2 Transfer Learning**, the model optimizes diagnostic accuracy while tracking **Recall** to minimize false negatives in medical assessment.

* **Dataset:** Kaggle - Chest X-Ray Images (Pneumonia)
* **Architectures:** Custom CNN & MobileNetV2 Transfer Learning
* **Primary Metrics:** Accuracy & Recall
* **Deployment:** Streamlit Web Application

---

## Performance & Results

| Architecture | Validation Accuracy | Validation Recall | Validation Loss |
| :--- | :--- | :--- | :--- |
| **Custom CNN** | **98.80%** | ~98.20% | ~0.0650 |
| **MobileNetV2 (Transfer Learning)** | 96.64% | **96.79%** | **0.0795** |

* **Loss Function:** `binary_crossentropy`
* **Output Activation:** `sigmoid`

---

## Key Features

* **Dual Architecture:** Evaluates a 5-layer custom CNN against pre-trained MobileNetV2 representations.
* **Clinical Focus:** Prioritizes **Recall (Sensitivity)** alongside Accuracy to avoid missing positive pneumonia cases.
* **Regularization:** Uses Dropout (0.4 - 0.5) to prevent overfitting.
* **Interactive UI:** Streamlit application for real-time image uploads and diagnostic predictions with confidence scores.

---

## Tech Stack

* **Python 3.x**
* **TensorFlow / Keras**
* **Streamlit**
* **OpenCV & Pillow**
* **NumPy, Matplotlib, Scikit-Learn**

---

## Dataset Structure

```text
chest_xray/
├── train/ (NORMAL, PNEUMONIA)
├── val/   (NORMAL, PNEUMONIA)
└── test/  (NORMAL, PNEUMONIA)
