import random
import numpy as np
from sklearn import svm


# Set random seed for consistent model behavior
np.random.seed(2018)
np.random.RandomState(2018)
random.seed(2018)


class SVMModel():

    def __init__(self, kernel="rbf", outlier_fraction=0.0001, gamma=0.5):
        """
        Initializes the SVM model for anomaly detection.

        Args:
            kernel (str, optional): Kernel function used by the SVM. Defaults to "rbf".
            outlier_fraction (float, optional): Fraction of data points considered outliers. Defaults to 0.0001 (0.01%).
            gamma (float, optional): Kernel coefficient for some kernels (e.g., rbf). Defaults to 0.5.
        """

        self.name = "ocsvm"
        self.model = svm.OneClassSVM(
            nu=outlier_fraction, kernel=kernel, gamma=gamma)

    def train(self, in_train, in_val):
        """
        Trains the SVM model on the provided training data.

        Args:
            in_train (numpy.ndarray): Training data.
        """

        self.model.fit(in_train)

    def compute_anomaly_score(self, df):
        """
        Computes anomaly scores for the given data points using the trained SVM model.

        Args:
            df (numpy.ndarray): Data for which to compute anomaly scores.

        Returns:
            numpy.ndarray: Anomaly scores for each data point in df.
        """

        preds = self.model.decision_function(df)
        preds = preds * -1  # Invert sign for easier interpretation (higher = more anomalous)

        return preds