import argparse
from models.ae import AutoencoderModel
from models.ocsvm import SVMModel
from models.vae import VAEModel
from utils import data_utils, eval_utils
import numpy as np


import logging
logging.basicConfig(level=logging.INFO)


def train_svm(in_train, in_test):
    """
    Trains a One-Class Support Vector Machine (SVM) for anomaly detection.

    Args:
        in_train (numpy.ndarray): Training data.
        in_test (numpy.ndarray): Testing data.

    Returns:
        dict: Dictionary containing evaluation metrics for the SVM model.
    """

    # Define default parameters for the SVM model
    svm_kwargs = {
        "kernel": "rbf",
        "gamma": 0.5,
        "outlier_frac": 0.0001
    }

    # Create an SVM model instance with the defined parameters
    svm = SVMModel(**svm_kwargs)

    # Train the SVM model on the training data
    svm.train(in_train, in_test)

    # Compute anomaly scores for both inlier (normal) and outlier data
    inlier_scores = svm.compute_anomaly_score(in_test)
    outlier_scores = svm.compute_anomaly_score(out_test)

    # Print the anomaly scores for inliers and outliers
    print(inlier_scores)
    print(outlier_scores)

    # Evaluate the model performance using the evaluation module
    metrics = eval_utils.evaluate_model(
        inlier_scores, outlier_scores, model_name="ocsvm", show_plot=False)
    print(metrics)

    # Return the evaluation metrics dictionary
    return metrics


def train_autoencoder(in_train, in_test):
    """
    Trains an Autoencoder model for anomaly detection.

    Args:
        in_train (numpy.ndarray): Training data.
        in_test (numpy.ndarray): Testing data.

    Returns:
        dict: Dictionary containing evaluation metrics for the Autoencoder model.
    """

    # Define default parameters for the Autoencoder model
    ae_kwargs = {
        "latent_dim": 2,
        "hidden_dim": [15, 7],
        "epochs": 14,
        "batch_size": 128
    }

    # Create an Autoencoder model instance with the defined parameters
    ae = AutoencoderModel(in_train.shape[1], **ae_kwargs)

    # Train the Autoencoder model on the training data
    ae.train(in_train, in_test)

    # Save the trained Autoencoder model for later use (optional)
    ae.save_model()

    # Compute anomaly scores for both inlier (normal) and outlier data
    inlier_scores = ae.compute_anomaly_score(in_test)
    outlier_scores = ae.compute_anomaly_score(out_test)

    # Print the anomaly scores for inliers and outliers
    print(inlier_scores)
    print(outlier_scores)

    # Evaluate the model performance using the evaluation module
    metrics = eval_utils.evaluate_model(
        inlier_scores, outlier_scores, model_name="ae", show_plot=False)
    print(metrics)

    # Return the evaluation metrics dictionary
    return metrics


def train_vae(in_train, in_test):
    """
    Trains a Variational Autoencoder (VAE) model for anomaly detection.

    Args:
        in_train (numpy.ndarray): Training data.
        in_test (numpy.ndarray): Testing data.

    Returns:
        dict: Dictionary containing evaluation metrics for the VAE model.
    """

    # Define default parameters for the VAE model (similar to Autoencoder)
    vae_kwargs = {
        "latent_dim": 2,
        "hidden_dim": [15, 7],
        "epochs": 8,
        "batch_size": 128
    }

    # Create a VAE model instance with the defined parameters
    vae = VAEModel(in_train.shape[1], **vae_kwargs)

    # Train the VAE model on the training data
    vae.train(in_train, in_test)

    # Save the trained VAE model for later use (optional)
    vae.save_model()

    inlier_scores = vae.compute_anomaly_score(in_test)
    outlier_scores = vae.compute_anomaly_score(out_test)
    print(inlier_scores)
    print(outlier_scores)
    metrics = eval_utils.evaluate_model(
        inlier_scores, outlier_scores, model_name="vae", show_plot=False)
    print(metrics)
    return metrics


def train_bigan(in_train, in_test):
    """
    Trains a Bidirectional Generative Adversarial Network (BiGAN) model for anomaly detection.

    Args:
        in_train (numpy.ndarray): Training data.
        in_test (numpy.ndarray): Testing data.

    Returns:
        dict: Dictionary containing evaluation metrics for the BiGAN model.
    """

    # Define default parameters for the BiGAN model
    bigan_kwargs = {
        "latent_dim": 2,
        "dense_dim": 128,
        "epochs": 15,
        "batch_size": 256,
        "learning_rate": 0.01
    }

    # Create a BiGAN model instance with the defined parameters
    input_shape = (in_train.shape[1], )
    bigan = BiGANModel(input_shape, **bigan_kwargs)

    # Train the BiGAN model on the training data
    bigan.train(in_train, in_test)

    # Save the trained BiGAN model for later use (optional)
    bigan.save_model()

    # Compute anomaly scores for both inlier (normal) and outlier data
    inlier_scores = bigan.compute_anomaly_score(in_test)
    outlier_scores = bigan.compute_anomaly_score(out_test)

    # Print the anomaly scores for inliers and outliers
    print(inlier_scores)
    print(outlier_scores)

    # Evaluate the model performance using the evaluation module
    metrics = eval_utils.evaluate_model(
        inlier_scores, outlier_scores, model_name="bigan", show_plot=False)
    print(metrics)

    # Return the evaluation metrics dictionary
    return metrics

def train_all():
    """
    Trains all the models (SVM, Autoencoder, VAE, and BiGAN) on the given data.
    """

    train_autoencoder()
    train_vae()
    train_svm()
    train_bigan()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process train parameters')
    parser.add_argument('-m', '--model', dest='model', type=str,
                        choices=["ae", "vae", "seq2seq", "gan", "all"],
                        help='model type to train', default="ae")

    args, unknown = parser.parse_known_args()

    # Load the KDD dataset
    test_data_partition = "8020"
    in_train, out_train, scaler, _ = data_utils.load_kdd(
        data_path="data/kdd/", dataset_type="train", partition=test_data_partition)
    in_test, out_test, _, _ = data_utils.load_kdd(
        data_path="data/kdd/", dataset_type="test", partition=test_data_partition, scaler=scaler)

    # Train the specified model
    if args.model == "ae":
        train_autoencoder()
    elif args.model == "vae":
        train_vae()
    elif args.model == "svm":
        train_svm()
    elif args.model == "bigan":
        train_bigan()
    elif args.model == "all":
        train_all()