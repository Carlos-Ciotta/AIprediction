from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np

def remove_outliers(data, threshold=1.5):
    #Q1 e Q3
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    inferior_limit = Q1 - threshold * IQR
    upper_limit = Q3 + threshold * IQR

    # Identifica as amostras que são outliers
    outliers = (data < inferior_limit) | (data > upper_limit)

    return outliers
#'Portugal'
def dataProcessing_toNumpy(data, test_rate):
    output = data['Active Energy (MWh) - Porto']
    y = np.array(output)
    input = data[[col for col in data.columns if col != 'Active Energy (MWh) - Porto']]
    x = np.array(input)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_rate, random_state=42)
    #x_train, y_train = shuffle(x_train, y_train, random_state=42)

    outliers_y = remove_outliers(y_train)
    # Remove outliers from samples
    x_train = x_train[~outliers_y]
    y_train = y_train[~outliers_y]

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    # Defining scaler
    scalerx_train = MinMaxScaler()
    scalerx_test = MinMaxScaler()
    scalery_train = MinMaxScaler()
    scalery_test = MinMaxScaler()

    # normalizing
    x_train = scalerx_train.fit_transform(x_train)
    x_test = scalerx_test.fit_transform(x_test)

    y_train = scalery_train.fit_transform(y_train)
    y_test = scalery_test.fit_transform(y_test)

    x_num = x.shape[1]
    steps = 1
    x_train = np.reshape(x_train, (x_train.shape[0],steps, x_num))
    x_test = np.reshape(x_test, (x_test.shape[0], steps,x_num))

    return x_train, x_test, y_train, y_test, scalerx_test, scalery_test

__all__ = [remove_outliers, dataProcessing_toNumpy]