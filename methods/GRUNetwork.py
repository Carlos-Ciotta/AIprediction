import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, Callback
import keras
from keras.models import load_model

class StopAtMAE(Callback):
    def __init__(self, target_mse):
        super(StopAtMAE, self).__init__()
        self.mse = target_mse  # Define o valor de MAE alvo

    def on_epoch_end(self, epoch, logs=None):
        # Obtém o valor de MAE a partir dos logs
        mse = logs.get('mse')
        if mse is not None and mse <= self.mse:
            print(f"\nMSE atingiu {mse}, que é menor ou igual ao alvo de {self.target_mse}. Interrompendo o treinamento.")
            self.model.stop_training = True  # Interrompe o treinamento

class GRUNetwork:
    def __init__(self,x_train,eta, epochs):
        self.model = Sequential()

        self.model.add(GRU(64, return_sequences=True, input_shape = (x_train.shape[1], x_train.shape[2]), activation ='hard_silu'))
        #self.model.add(Dropout(0.2))
        self.model.add(GRU(32, return_sequences=False, activation ='hard_silu'))
        #self.model.add(Dropout(0.2))
        self.model.add(Dense(1, activation = 'hard_silu'))

        self.optimizer = Adam(learning_rate=eta)
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=int(400), restore_best_weights=True)

        self.model.compile(optimizer=self.optimizer, loss='mean_squared_error', metrics = ['mse'])

    def train(self, x_train, y_train, epochs, batch_size, x_test, y_test):
       # mse_callback = StopAtMAE(target_mse=0.009)
        self.model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), batch_size=batch_size, callbacks = [self.early_stopping])

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def predict(self, x_test, y_test, scaler_y):
        y_pred = self.model.predict(x_test)
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1))
        y_test = scaler_y.inverse_transform(y_test.reshape(-1,1))

        return y_pred, y_test

    def save_model(self, filename):
        self.model.save(filename)

    def load_model(self, filename):
        self.model = load_model(filename)

    def plot_prediction(self,y_pred, y_test):
        # Figure Size
        plt.figure(figsize=(10, 6))

        # Ploting lines
        plt.plot(y_test[:72,0], label="Energy Consumption", color='blue', linestyle='dashed')
        plt.plot(y_pred[:72, 0], label="Prediction", color='red', alpha=0.7)

        # Title and Label
        plt.title("Real values x Predicted vales", fontsize=14)
        plt.xlabel("Hours", fontsize=12)
        plt.ylabel("Energetic Balance", fontsize=12)

        # Ading legends of lines
        plt.legend()

        # Show Graph
        return plt.show()

__all__ = ['GRUNetwork']