# Importing necessary libraries
from sklearn.preprocessing import MinMaxScaler  # For scaling data to a specific range (normalization)
from sklearn.utils import shuffle  # For shuffling data to randomize its order
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
import numpy as np  # For working with arrays and numerical operations

# Function to identify outliers in a dataset using the interquartile range (IQR)
def remove_outliers(data, threshold=1.5):
    # Calculate the first quartile (Q1) and third quartile (Q3) of the data
    Q1 = np.percentile(data, 25)  # 25th percentile
    Q3 = np.percentile(data, 75)  # 75th percentile

    # Compute the interquartile range (IQR)
    IQR = Q3 - Q1

    # Determine the lower and upper bounds for outliers
    inferior_limit = Q1 - threshold * IQR
    upper_limit = Q3 + threshold * IQR

    # Identify data points that are outside the bounds (outliers)
    outliers = (data < inferior_limit) | (data > upper_limit)

    return outliers  # Returns a boolean array marking outliers

# Function to process a dataset and prepare it for use in machine learning models
def dataProcessing_toNumpy(data, test_rate, out):
    # Extract the output (target variable) column from the dataset
    output = data[out]
    y = np.array(output)  # Convert the output column to a NumPy array

    # Extract input (feature) columns, excluding the target variable
    input = data[[col for col in data.columns if col != out]]
    x = np.array(input)  # Convert input data to a NumPy array

    # Split the dataset into training and testing sets based on the specified test rate
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_rate, random_state=42)

    # Identify outliers in the training target variable (y_train)
    outliers_y = remove_outliers(y_train)

    # Remove the identified outliers from the training dataset
    x_train = x_train[~outliers_y]  # Exclude rows corresponding to outliers in x_train
    y_train = y_train[~outliers_y]  # Exclude corresponding rows in y_train

    # Reshape the target variables into 2D arrays as required by certain ML models
    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    # Define scalers for normalizing both features and target variables
    scalerx_train = MinMaxScaler()  # Scaler for normalizing x_train
    scalerx_test = MinMaxScaler()  # Scaler for normalizing x_test
    scalery_train = MinMaxScaler()  # Scaler for normalizing y_train
    scalery_test = MinMaxScaler()  # Scaler for normalizing y_test

    # Normalize the input (features) for both training and testing sets
    x_train = scalerx_train.fit_transform(x_train)
    x_test = scalerx_test.fit_transform(x_test)

    # Normalize the target variables for both training and testing sets
    y_train = scalery_train.fit_transform(y_train)
    y_test = scalery_test.fit_transform(y_test)

    # Reshape input variables to be 3D arrays suitable for sequential models (e.g., RNNs, GRUs, LSTMs)
    x_num = x.shape[1]  # Number of features in the dataset
    steps = 1  # Number of time steps (1 in this case)
    x_train = np.reshape(x_train, (x_train.shape[0], steps, x_num))  # Reshape x_train to (samples, steps, features)
    x_test = np.reshape(x_test, (x_test.shape[0], steps, x_num))  # Reshape x_test similarly

    # Return the processed datasets and the scalers used for future transformations
    return x_train, x_test, y_train, y_test, scalerx_test, scalery_test

# Specify which functions are part of the public API for this script/module
__all__ = [remove_outliers, dataProcessing_toNumpy]
