from methods.GRUNetwork import GRUNetwork
import data_processing.DatasetProcessing as dp
import data_processing.NumpyProcessingDL as tnp
import os
import pandas as pd

eta = 1e-2
batch_size = 32
epochs = 2000
test_rate = 0.2

features_consumption = ['solarradiation',
                'humidity','month','hour','Clear','Overcast','Partially cloudy',"Rain, Partially cloudy",
                'Domingo','Quarta','Quinta','Segunda','Sexta','Sábado','Terça','Active Energy (MWh) - Porto'
                ,'Portugal']

if __name__ == "__main__":
        if(os.path.exists('datasets/data_training.csv')):
            data = pd.read_csv('datasets/data_training.csv')
            x_train,x_test, y_train, y_test, scalerx, scalery = tnp.dataProcessing_toNumpy(data[features_consumption], test_rate)
            rnn = GRUNetwork(x_train,eta,epochs)

        else:
            dp.merged_processing('datasets/posta_code_energy_consumption_porto2023.csv', 
                                 'datasets/weather_porto 2023-01-01 to 2023-09-30.csv',
                                  'datasets/Porto_holidays.txt',
                                  'datasets/prices_energy_20230101_20240101.csv',
                                  'datasets/energy_production_portugal_20230101_20240101.csv')
            data = pd.read_csv('datasets/data_training.csv')
            x_train,x_test, y_train, y_test, scalerx, scalery = tnp.dataProcessing_toNumpy(data[features_consumption], test_rate)
            rnn = GRUNetwork(x_train,eta,epochs)