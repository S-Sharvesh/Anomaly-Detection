import numpy as np
import pandas as pd
from utils import data_utils

# Load KDD dataset, partitioning data into training and testing sets
test_data_partition = "8020"  # 80% for training, 20% for testing
in_train, out_train, scaler, col_names = data_utils.load_kdd(
    data_path="data/kdd/", dataset_type="train", partition=test_data_partition)
in_test, out_test, _, _ = data_utils.load_kdd(
    data_path="data/kdd/", dataset_type="test", partition=test_data_partition, scaler=scaler)

# Define the number of inliers and outliers to include in the response
inlier_size = 2
outlier_size = 2

def data_to_json(data, label):
    """
    Converts data and labels into a JSON-compatible format.

    Args:
        data (numpy.ndarray): The input data.
        label (int): The label for the data points (0 for inliers, 1 for outliers).

    Returns:
        pandas.DataFrame: A DataFrame containing the data and labels, suitable for JSON conversion.
    """

    data = pd.DataFrame(data, columns=list(col_names))  # Create a DataFrame with column names
    data["label"] = label  # Add a column for the label
    return data

# Sample inliers and outliers from the test data
in_liers = data_to_json(
    in_test[np.random.randint(5, size=inlier_size), :], 0)  # Randomly select 2 inliers
out_liers = data_to_json(
    out_test[np.random.randint(5, size=outlier_size), :], 1)  # Randomly select 2 outliers

# Concatenate inliers and outliers into a single DataFrame
response = pd.concat([in_liers, out_liers], axis=0)

# Convert the DataFrame to JSON format and print it
print(response.to_json(orient="records"))