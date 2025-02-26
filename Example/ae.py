import tensorflow
from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
import logging

import os
from utils import train_utils  # Assuming this contains training utility functions

import numpy as np
import random


# Set random seed for consistent model behavior
tensorflow.random.set_seed(2018)
np.random.seed(2018)
np.random.RandomState(2018)
random.seed(2018)


class AutoencoderModel():

    def __init__(self, n_features, hidden_layers=2, latent_dim=2, hidden_dim=[15, 7],
                 output_activation='sigmoid', learning_rate=0.01, epochs=15, batch_size=128, model_path=None):
        """
        This function initializes the Autoencoder model.

        Args:
            n_features (int): Number of features in the input data.
            hidden_layers (int, optional): Number of hidden layers used in encoder and decoder. Defaults to 2.
            latent_dim (int, optional): Dimensionality of the latent representation (compressed data). Defaults to 2.
            hidden_dim (list, optional): List specifying the number of units in each hidden layer. Defaults to [15, 7].
            output_activation (str, optional): Activation function for the last layer of the decoder. Defaults to 'sigmoid'.
            learning_rate (float, optional): Learning rate used during training. Defaults to 0.01.
            epochs (int, optional): Number of training epochs. Defaults to 15.
            batch_size (int, optional): Batch size used during training. Defaults to 128.
            model_path (str, optional): Path to load a pre-trained model from. Defaults to None.
        """

        self.epochs = epochs
        self.batch_size = batch_size
        self.model_name = "ae"

        self.create_model(n_features, hidden_layers=hidden_layers, latent_dim=latent_dim,
                          hidden_dim=hidden_dim, output_activation=output_activation,
                          learning_rate=learning_rate, model_path=model_path)

    def create_model(self, n_features, hidden_layers=1, latent_dim=2, hidden_dim=[],
                     output_activation='sigmoid', learning_rate=0.001, model_path=None):

        # Define hidden layer dimensions if not provided
        if hidden_dim == []:
            current_dim = n_features
            for _ in range(hidden_layers):
                hidden_dim.append(int(max([current_dim / 2, 2])))  # Ensure at least 2 units per layer
                current_dim /= 2

        # Optional: L1/L2 regularization to prevent overfitting (commented out for now)
        # kernel_regularizer = regularizers.l1_l2(l1=0.01, l2=0.01)
        # kernel_regularizer = regularizers.l1(0.01)
        kernel_regularizer = None

        # Build the Autoencoder (encoder + decoder)

        # Encoder
        inputs = Input(shape=(n_features,), name='encoder_input')
        encoded = Dense(hidden_dim[0], activation='relu',