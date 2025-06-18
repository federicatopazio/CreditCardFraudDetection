# 🛡️ Credit Card Fraud Detection using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This project reproduces the results of a published study on credit card fraud detection using deep learning methods. It evaluates the performance of 1D CNNs, 2D CNNs, and LSTM networks on two real-world imbalanced datasets.

> **Nguyen, Thanh Thi, et al.**  
> *Deep Learning Methods for Credit Card Fraud Detection.*  
> arXiv preprint [arXiv:2012.03754](https://arxiv.org/abs/2012.03754), 2020.

---

## 📁 Project Structure
├── data/ # Dynamic retrieval from Kaggle
├── notebooks/
│ └── fraud_detection.ipynb
├── Report.pdf
└── README.md


---

## 🧪 Datasets

- **European Card Data (ECD)**  
  [Kaggle link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
  284,807 samples, 31 features (0.172% fraud)

- **Small Card Data (SCD)**  
  [Kaggle link](https://www.kaggle.com/datasets/shubhamjoshi2130of/abstract-data-set-for-credit-card-fraud-detection)  
  3,075 samples, 12 features (14.6% fraud)

---

## ⚙️ Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-naml.git
   cd credit-card-fraud-naml
   ```
2. **Add your Kaggle credentials**
Save your kaggle.json token inside a .kaggle/ directory:
  ```bash
  mkdir ~/.kaggle
  cp path/to/kaggle.json ~/.kaggle/
  chmod 600 ~/.kaggle/kaggle.json
  ```
3. **Run the notebook**
Open and run the main notebook:
  ```bash
  jupyter notebook notebooks/fraud_detection.ipynb
  ```
## 🧠 Models

This project implements three neural network architectures, each evaluated on both the European Card Data (ECD) and Small Card Data (SCD) datasets. All models use binary cross-entropy loss and the Adam optimizer.

### 📌 1D CNN

Designed for structured, sequential data like time series

### 🧩 2D CNN

Used after reshaping tabular data into 2D matrices (e.g., 5×6) to simulate spatial structure

### 🔁 LSTM

Captures temporal dependencies and long-term patterns in transaction sequences

Each model was trained on balanced and imbalanced versions of the datasets using:

- Random Under-Sampling
- NearMiss
- SMOTE

Evaluation metrics include accuracy, precision, recall, and F1 score. EarlyStopping and learning rate callbacks were also used to improve generalization and avoid overfitting.
