# Deep Learning for Anomaly Detection

This repository contains experimental code to implement deep learning techniques for anomaly detection. It also launches an interactive dashboard to visualize model results applied to a network intrusion use case. 

The repository includes implementations of several neural networks (Autoencoder, Variational Autoencoder, Bidirectional GAN, Sequence Models) in TensorFlow 2.0, along with two other baselines (One-Class SVM, PCA).

![Anomaly Detection](https://github.com/user-attachments/assets/c85acbd3-5101-4ff6-a9ae-39f8d48ce996)

## Introduction

Anomalies—often referred to as outliers, abnormalities, rare events, or deviants—are data points that do not conform to expected behavior. Anomaly detection involves identifying these patterns in data that deviate from normal trends. 

Detecting anomalies is critical in various industries, including:
- IT analytics
- Network intrusion detection
- Medical diagnostics
- Financial fraud protection
- Manufacturing quality control
- Marketing and social media analytics

## How Anomaly Detection Works

Most anomaly detection methods follow this approach:
1. **Model Normal Behavior:** Train the model on normal data, assuming anomalies are rare.
2. **Assign Anomaly Scores:** Measure how much a data point deviates from normal behavior.
3. **Thresholding:** Define a threshold to classify a data point as normal or anomalous.

In this repository, models use a **reconstruction error approach**—i.e., a model is trained to reconstruct input data. At test time:
- **Normal samples** are reconstructed with low error.
- **Anomalous samples** result in higher reconstruction errors.

For example, an autoencoder learns to reconstruct normal data. If a test sample has high reconstruction error, it is flagged as an anomaly.

## Repository Structure

```
├── data
│   ├── kdd
│   ├── kdd_data_gen.py
├── cml
│   ├── install_deps.py
├── metrics
├── models
│   ├── ae.py
│   ├── bigan.py
│   ├── ocsvm.py
│   ├── pca.py
│   ├── seq2seq.py
│   ├── vae.py
├── utils
│   ├── data_utils.py
│   ├── eval_utils.py
│   ├── train_utils.py
├── train.py
├── test.py
```

### **data/**
Holds the **KDD Network Intrusion dataset** used in experiments and the interactive dashboard. The script `kdd_data_gen.py` downloads and preprocesses the data.

### **cml/**
Contains artifacts needed to configure and launch the project on **Cloudera Machine Learning (CML)**.

### **models/**
Contains implementations of various anomaly detection models:
- **ae.py** - Autoencoder
- **bigan.py** - Bidirectional GAN
- **ocsvm.py** - One-Class SVM
- **pca.py** - Principal Component Analysis
- **seq2seq.py** - Sequence Models
- **vae.py** - Variational Autoencoder

### **utils/**
Holds helper functions used across different scripts, including data preprocessing, evaluation, and training utilities.

### **train.py**
This script trains and evaluates the models. Steps:
1. Download the KDD dataset (if not already downloaded).
2. Train multiple models: **Autoencoder, Variational Autoencoder, Bidirectional GAN, Sequence Models, PCA, OCSVM**.
3. Evaluate models on a test split (**8000 inliers, 2000 outliers**).
4. Generate performance metrics:
   - Histogram of anomaly scores
   - ROC curves
   - General evaluation metrics (F1-score, Precision, Recall, Accuracy)

## Summary of Results

![Model Results](https://github.com/user-attachments/assets/629640f6-c274-4ba4-aed8-425430dcb4ab)

Each model was evaluated using labeled test data, selecting an optimal threshold to maximize accuracy. We report:
- **F1, F2 scores**
- **Precision, Recall**
- **ROC (Area Under Curve)**

**Findings:**
- **Deep learning models (BiGAN, AE) performed well**, achieving higher precision and recall compared to PCA and One-Class SVM.
- **Sequence-to-sequence models were less effective**, as the dataset is not temporal.
- On **more complex datasets (e.g., images)**, deep learning models would likely show even stronger advantages.

For more details, refer to the **report**.

## Choosing a Modeling Approach

Different deep learning methods have unique strengths:
- **Sequence-based data:** Use **sequence-to-sequence models** (LSTMs, GRUs).
- **Uncertainty estimation:** Use **Variational Autoencoders (VAE)** or **GAN-based models**.
- **Image data:** Use **Autoencoders, VAEs, or GANs** with convolutional layers.

Each model is evaluated using precision, recall, F1-score, and ROC AUC. Deep models like BiGAN and Autoencoder perform better than traditional baselines (PCA, One-Class SVM), especially on complex datasets.

## How to Decide on a Modeling Approach?

Different models are suited for different types of data. The table below summarizes when to use each model:

| Model | Pros | Cons |
|-------|------|------|
| **AutoEncoder** | Learns complex non-linear patterns | Requires a large dataset for training, does not estimate uncertainty |
| **Variational AutoEncoder (VAE)** | Provides uncertainty estimates (probabilistic measure) | Training can be slow and requires a large dataset |
| **GAN (BiGAN)** | Learns complex distributions, robust in semi-supervised learning | Training instability (mode collapse), requires large data & long training time |
| **Sequence-to-Sequence Model** | Effective for temporal data | Slow inference, less effective for non-sequential data |
| **One-Class SVM** | Fast training & inference, requires less data | Limited in capturing complex relationships, requires careful parameter tuning |

For instance:
- Use **Sequence Models (LSTM, Seq2Seq)** for time-series or sequential data.
- Use **VAE or GAN** when uncertainty estimation is important.
- Use **Autoencoders** for feature-based anomaly detection in structured data.
- Use **PCA or One-Class SVM** for quick, traditional anomaly detection on small datasets.

## Conclusion

This repository provides a strong foundation for deep learning-based anomaly detection, supporting multiple modeling approaches. The interactive dashboard allows visualization of model performance across datasets.

For more details, check out our report! 🚀

![Anomaly Visualization](https://github.com/user-attachments/assets/ad736b84-8dae-4437-ae93-619ffb04b82f)

## License
This project is licensed under the **MIT License**.

---
