# Importing the required libraries for the implementation
import matplotlib.pyplot as plt  # Library for creating plots and visualizations
from keras.models import Sequential  # For building sequential neural network models
from keras.layers import Dense, Dropout, GRU  # For adding layers such as GRU, Dense, and Dropout
from keras.optimizers import Adam  # Adam optimizer for gradient-based optimization
from keras.callbacks import EarlyStopping, Callback  # Callbacks for controlling training processes
import keras  # Core Keras library
from keras.models import load_model  # For saving and loading trained models

# Class defining a GRU-based neural network model
class GRUNetwork:
    def __init__(self, x_train, eta, epochs):
        # Initializes the sequential model
        self.model = Sequential()

        # Adds the first GRU layer with 64 units, returning sequences for further layers
        self.model.add(GRU(64, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2]), activation='hard_silu'))
        
        # Optionally, dropout layers can be added (currently commented out)
        # self.model.add(Dropout(0.2))
        
        # Adds a second GRU layer with 32 units, outputting the final sequence
        self.model.add(GRU(32, return_sequences=False, activation='hard_silu'))
        
        # self.model.add(Dropout(0.2))  # Another optional dropout layer (commented out)

        # Adds a Dense layer to output a single value (e.g., regression output)
        self.model.add(Dense(1, activation='hard_silu'))

        # Sets up the Adam optimizer with the specified learning rate
        self.optimizer = Adam(learning_rate=eta)

        # Implements early stopping to prevent overfitting; monitors validation loss
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=400, restore_best_weights=True)

        # Compiles the model with Mean Squared Error loss and MSE as a metric
        self.model.compile(optimizer=self.optimizer, loss='mean_squared_error', metrics=['mse'])

    # Method to train the model
    def train(self, x_train, y_train, epochs, batch_size, x_test, y_test):
        # Trains the model with training and validation data; uses early stopping
        self.model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), 
                       batch_size=batch_size, callbacks=[self.early_stopping])

    # Method to evaluate the model on the test dataset
    def evaluate(self, y_pred, y_test):
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y_test, y_pred)
        # Calculating the Mean Squared Error (MSE) to evaluate the model's prediction error.
        r2 = r2_score(y_test, y_pred)
        # Calculating the R-squared (R²) to measure the goodness of fit of the model.
        return mse, r2

    # Method to make predictions and scale them back to original values
    def predict(self, x_test, y_test, scaler_y):
        # Generates predictions on the test dataset
        y_pred = self.model.predict(x_test)
        # Inverses the scaling of predictions and ground truth
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
        y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1))

        return y_pred, y_test

    # Method to save the trained model to a file
    def save_model(self, filename):
        self.model.save(filename)

    # Method to load a previously saved model from a file
    def load_model(self, filename):
        self.model = load_model(filename)

    # Method to visualize the predicted values against the actual values
    def plot_prediction(self, y_pred, y_test, type):
        plt.figure(figsize=(10, 6))

        # Plots the predicted values as a red line
        plt.plot(y_pred[:72, 0], label="Prediction", color='red', alpha=0.7)
        plt.plot(y_test[:72, 0], label="Real Value", color='blue', alpha=0.7)
        if(type == 'consumption'):
            # Adds title and axis labels
            plt.title("Energy Consumption", fontsize=14)
            plt.xlabel("Hours", fontsize=12)
            plt.ylabel("Energy Consumption (MWh)", fontsize=12)
        elif(type=='price'):
            plt.title("Energy Price", fontsize=14)
            plt.xlabel("Hours", fontsize=12)
            plt.ylabel("Price (Euro/MWh)", fontsize=12)

        # Displays a legend for the plot
        plt.legend()

        # Shows the plot
        return plt.show()

# Defines the public API for this script/module
__all__ = ['GRUNetwork']
