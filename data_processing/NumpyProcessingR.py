# Importing necessary libraries
from sklearn.preprocessing import MinMaxScaler  # For scaling data to a specified range (normalization)
from sklearn.utils import shuffle  # To randomly shuffle datasets
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
import numpy as np  # For numerical operations on arrays

# Function to identify outliers in a dataset based on the interquartile range (IQR)
def remove_outliers(data, threshold=1.5):
    # Calculate the first quartile (Q1) and third quartile (Q3)
    Q1 = np.percentile(data, 25)  # 25th percentile
    Q3 = np.percentile(data, 75)  # 75th percentile

    # Compute the interquartile range (IQR)
    IQR = Q3 - Q1

    # Define the lower and upper limits for identifying outliers
    inferior_limit = Q1 - threshold * IQR
    upper_limit = Q3 + threshold * IQR

    # Identify samples that are outliers (values outside the IQR bounds)
    outliers = (data < inferior_limit) | (data > upper_limit)

    return outliers  # Returns a boolean mask indicating outlier samples

# Function to process and prepare data for use in machine learning models
def dataProcessing_toNumpy(x, y, test_rate):
    # Convert input datasets x and y into NumPy arrays for easier manipulation
    x = np.array(x)
    y = np.array(y)

    # Split the data into training and testing sets based on the specified test rate
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_rate)

    # Shuffle the training data to ensure randomness and avoid biases
    x_train, y_train = shuffle(x_train, y_train, random_state=42)

    # Identify outliers in the target variable (y_train)
    outliers_y = remove_outliers(y_train)

    # Remove the identified outliers from the training dataset
    x_train = x_train[~outliers_y]  # Exclude rows corresponding to outliers
    y_train = y_train[~outliers_y]

    # Reshape y_train and y_test to be 2D arrays, as required by many machine learning models
    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    # Define scalers for normalizing the data
    scalerx_train = MinMaxScaler()  # Scaler for x_train
    scalerx_test = MinMaxScaler()  # Scaler for x_test
    scalery_train = MinMaxScaler()  # Scaler for y_train
    scalery_test = MinMaxScaler()  # Scaler for y_test

    # Normalize the training and testing feature data (x_train and x_test)
    x_train = scalerx_train.fit_transform(x_train)
    x_test = scalerx_test.fit_transform(x_test)

    # Normalize the training and testing target data (y_train and y_test)
    y_train = scalery_train.fit_transform(y_train)
    y_test = scalery_test.fit_transform(y_test)

    # Return the processed datasets and scalers
    return x_train, x_test, y_train, y_test, scalerx_test, scalery_test

# Specify which functions are part of the public API for this script/module
__all__ = [remove_outliers, dataProcessing_toNumpy]
