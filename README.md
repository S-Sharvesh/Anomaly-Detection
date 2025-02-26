# Deep Learning for Anomaly Detection

This repo contains experimental code used to implement deep learning techniques for the task of anomaly detection and launches an interactive dashboard to visualize model results applied to a network intrusion use case. We include implementations of several neural networks (Autoencoder, Variational Autoencoder, Bidirectional GAN, Sequence Models) in Tensorflow 2.0 and two other baselines (One Class SVM, PCA).

For an in-depth review of the concepts presented here, please consult the Cloudera Fast Forward report Deep Learning for Anomaly Detection. Additionally, two related prototypes are available for reference - Blip & Anomagram

AutoEncoder	Variational AutoEncoder	BiGAN
		
Seq2Seq	PCA	OCSVM
		
Anomalies - often referred to as outliers, abnormalities, rare events, or deviants - are data points or patterns in data that do not conform to a notion of normal behavior. Anomaly detection, then, is the task of finding those patterns in data that do not adhere to expected norms, given previous observations. The capability to recognize or detect anomalous behavior can provide highly useful insights across industries. Flagging unusual cases or enacting a planned response when they occur can save businesses time, costs, and customers. Hence, anomaly detection has found diverse applications in a variety of domains, including IT analytics, network intrusion analytics, medical diagnostics, financial fraud protection, manufacturing quality control, marketing and social media analytics, and more.

How Anomaly Detection Works

The underlying strategy for most approaches to anomaly detection is to first model normal behavior, and then exploit this knowledge to identify deviations (anomalies). In this repo, the process includes the following steps:

Build a model of normal behavior using available data. Typically the model is trained on normal behavior data or assumes a small amount of abnormal data.
Based on this model, assign an anomaly score to each data point that represents a measure of deviation from normal behavior. The models in this repo use a reconstruction error approach, where the model attempts to reconstruct a sample at test time, and uses the reconstruction error as an anomaly score. The intuition is that normal samples will be reconstructed with almost no error, while abnormal/unusual samples will be reconstructed with larger error margins.
Apply a threshold on the anomaly score to determine which samples are anomalies.
As an illustrative example, an autoencoder model is trained on normal samples where the task is to reconstruct the input. At test time, we can use the reconstruction error (mean squared error) for each sample as anomaly scores.

Structure of Repo

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
data

The data directory holds the KDD Network Intrusion dataset used the experiments and interactive dashboard. It contains a script (kdd_data_gen.py) that downloads the data, constructs train and test sets separated into inliers and outliers, and places those data files in the data/kdd directory.

cml

The cml folder contains the artifacts needed to configure and launch the project on Cloudera Machine Learning (CML).

models

The models directory contains modules for each of the model implementations. Each module comes with code to specify parameters and methods for training and computing an anomaly score. It also serves as the holding location of saved models after training.

utils

The utils directory holds helper functions that are referenced throughout different modules and scripts in the repo.

train.py

This script contains code to train and then evaluate each model by generating a histogram of anomaly scores assigned by each model, and ROC curve to assess model skill on the anomaly detection task. The general steps taken in the script are:

Download the KDD dataset if not already downloaded
Trains all of the models (Autoencoder, Variational Autoencoder, Bidirectional GAN, Sequence Models, PCA, OCSVM)
Evaluates the models on a test split (8000 inliers, 2000 outliers). Generates charts on model performance: histogram of anomaly scores, ROC, general metrics (f1,f2, precision, recall, accuracy)
Summary of Results

AutoEncoder	Variational AutoEncoder	BiGAN
		
Seq2Seq	PCA	OCSVM
		
For each model, we use labeled test data to first select a threshold that yields the best accuracy and then report on metrics such as f1, f2, precision, and recall at that threshold. We also report on ROC (area under the curve) to evaluate the overall skill of each model. Given that the dataset we use is not extremely complex (18 features), we see that most models perform relatively well. Deep models (BiGAN, AE) are more robust (precision, recall, ROC AUC), compared to PCA and OCSVM. The sequence-to-sequence model is not particularly competitive, given the data is not temporal. On a more complex dataset (e.g., images), we expect to see (similar to existing research), more pronounced advantages in using a deep learning model.

For additional details on each model, see our report. Note that models implemented here are optimized for tabular data. For example, extending this to work with image data will usually require the use of convolutional layers (as opposed to dense layers) within the neural network models to achieve performant results.

AutoEncoder	Variational AutoEncoder	BiGAN
		
Seq2Seq	PCA	OCSVM
		
How to Decide on a Modeling Approach?

Given the differences between the deep learning methods discussed above (and their variants), it can be challenging to decide on the right model. When data contains sequences with temporal dependencies, a sequence-to-sequence model (or architectures with LSTM layers) can model these relationships well, yielding better results. For scenarios requiring principled estimates of uncertainty, generative models such as a VAE and GAN based approaches are suitable. For scenarios where the data is images, AEs, VAEs and GANs designed with convolution layers are suitable. The following table highlights the pros and cons of the different types of models, to provide guidance on when they are a good fit.

Model	Pros	Cons
AutoEncoder	
Flexible approach to modeling complex non-linear patterns in data
Does not support variational inference (estimates of uncertainty)
Requires a large dataset for training
Variational AutoEncoder	
Supports variational inference (probabilistic measure of uncertainty)
Requires a large amount of training data, training can take a while
GAN (BiGAN)	
Supports variational inference (probabilistic measure of uncertainty)
Use of discriminator signal allows better learning of data manifold Mihaela Rosca (2018), Issues with VAEs and GANs, CVPR18(useful for high dimensional image data).
GANs trained in semi-supervised learning mode have shown great promise, even with very few labeled data Raghavendra Chalapathy et al. (2019) "Deep Learning for Anomaly Detection: A Survey"
Requires a large amount of training data, and longer training time (epochs) to arrive at stable results Tim Salimans et al. (2016) "Improved Techniques for Training GANs", Neurips 2016
Training can be unstable (GAN mode collapse)
Sequence-to-Sequence Model	
Well suited for data with temporal components (e.g., discretized time series data)
Slow inference (compute scales with sequence length which needs to be fixed)
Training can be slow
Limited accuracy when data contains features with no temporal dependence
Supports variational inference (probabilistic measure of uncertainty)
One Class SVM	
Does not require a large amount of data
Fast to train
Fast inference time
Limited capacity in capturing complex relationships within data
Requires careful parameter selection (kernel, nu, gamma) that need to be carefully tuned.
Does not model a probability distribution, harder to compute estimates of confidence.
