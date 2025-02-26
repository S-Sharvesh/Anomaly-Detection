import tensorflow
from tensorflow.keras.layers import Lambda, Input, Dense, Activation, Flatten, concatenate, Reshape, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
import logging
from tensorflow.keras.layers import LeakyReLU, ReLU
import os


import numpy as np
import random


# Set random seed for consistent model behavior
tensorflow.random.set_seed(2018)
np.random.seed(2018)
np.random.RandomState(2018)
random.seed(2018)


class BiGANModel():

    def __init__(self,  input_shape, dense_dim=64, latent_dim=32,
                 output_activation='sigmoid', learning_rate=0.01, epochs=15, batch_size=128, model_path=None):
        """
        Initializes the BiGAN model.

        Args:
            input_shape (tuple): Shape of the input data.
            dense_dim (int, optional): Number of units in the dense layers of the model. Defaults to 64.
            latent_dim (int, optional): Dimensionality of the latent representation. Defaults to 32.
            output_activation (str, optional): Activation function for the last layer of the generator. Defaults to 'sigmoid'.
            learning_rate (float, optional): Learning rate used during training. Defaults to 0.01.
            epochs (int, optional): Number of training epochs. Defaults to 15.
            batch_size (int, optional): Batch size used during training. Defaults to 128.
            model_path (str, optional): Path to load a pre-trained model from. Defaults to None.
        """

        self.name = "bigan"
        self.epochs = epochs
        self.dense_dim = dense_dim
        self.batch_size = batch_size
        self.latent_dim = latent_dim

        self.create_model(input_shape=input_shape,
                          learning_rate=learning_rate)

    def create_model(self, input_shape=(18, 1, 1),   learning_rate=0.01):
        """
        Defines the architecture of the BiGAN model.

        Args:
            input_shape (tuple): Shape of the input data. Defaults to (18, 1, 1).
            learning_rate (float, optional): Learning rate used during training. Defaults to 0.01.
        """

        self.input_shape = input_shape

        # Adam optimizer with learning rate
        optimizer = Adam(lr=learning_rate)

        # Build and compile the discriminator model
        self.discriminator = self.build_discriminator()
        logging.info(self.discriminator.summary())
        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator model
        self.generator = self.build_generator()
        logging.info(self.generator.summary())

        # Build the encoder model
        self.encoder = self.build_encoder()
        logging.info(self.encoder.summary())

        # During combined training, set the discriminator to non-trainable
        self.discriminator.trainable = False

        # Input noise for generator
        z = Input(shape=(self.latent_dim, ), name="inputnoise")
        input_data_ = self.generator(z)

        # Input data for encoder
        input_data = Input(shape=self.input_shape,  name="inputimage")
        z_ = self.encoder(input_data)

        # Discriminator treats generated data (latent -> input) as fake and real data (input -> latent) as valid
        fake = self.discriminator([z, input_data_])
        valid = self.discriminator([z_, input_data])

        # Combined model (trains generator to fool discriminator)
        self.bigan_generator = Model(
            [z, input_data], [fake, valid], name="bigan")
        self.bigan_generator.compile(
            loss=['binary_crossentropy', 'binary_crossentropy'], optimizer=optimizer)
        logging.