from methods.GRUNetwork import GRUNetwork
import data_processing.DatasetProcessing as dp
import data_processing.NumpyProcessingDL as tnp
import os
import pandas as pd


eta = 1e-2
batch_size = 32
epochs = 600
test_rate = 0.2
features_consumption = ['solarradiation','humidity','holiday'
                        ,'month','hour','Clear','Overcast','Partially cloudy',"Rain, Partially cloudy",
                        'Domingo','Quarta','Quinta','Segunda','Sexta','Sábado','Terça'
                        ,'Portugal','Active Energy (MWh) - Porto']
                        
price_pred = ['solarradiation','Hídrica','Eólica','Solar',
'Gás Natural - Ciclo Combinado','Gás natural - Cogeração','Importação',
'Bombagem','Portugal'
,'Active Energy (MWh) - Porto']

if __name__ == "__main__":
    try:
        if(os.path.exists('datasets/data_training.csv')):
            data = pd.read_csv('datasets/data_training.csv')
            x_train,x_test, y_train, y_test, scalerx, scalery = tnp.dataProcessing_toNumpy(data[features_consumption], test_rate)
            rnn = GRUNetwork(x_train,eta,epochs)

        else:
            dp.merged_processing('datasets/posta_code_energy_consumption_porto2023.csv', 
                                 'datasets/weather_porto 2022-08-30 to 2023-09-30.csv',
                                  'datasets/Porto_holidays.txt',
                                  'datasets/prices_energy_20220101_20241209.csv',
                                  'datasets/energy_production_portugal_20220101_20240912.csv')
            data = pd.read_csv('datasets/data_training.csv')
            x_train,x_test, y_train, y_test, scalerx, scalery = tnp.dataProcessing_toNumpy(data[features_consumption], test_rate)
            rnn = GRUNetwork(x_train,eta,epochs)
    except:
        print('Erros')

    else:
        if(os.path.exists('RNNmodels/cosumption_model4.h5')):
            rnn.load_model('RNNmodels/cosumption_model4.h5')
            loss, accuracy = rnn.evaluate(x_test, y_test)
            print(f'loss : {loss}\naccuracy : {accuracy}')
            y_pred, y_test = rnn.predict(x_test, y_test, scalery)
            rnn.plot_prediction(y_pred, y_test)
        else:
            rnn.train(x_train, y_train, epochs, batch_size, x_test,y_test)
            loss, accuracy = rnn.evaluate(x_test, y_test)
            print(f'loss : {loss}\naccuracy : {accuracy}')
            rnn.save_model('cosumption_model5.h5')
            y_pred, y_test = rnn.predict(x_test, y_test, scalery)
            rnn.plot_prediction(y_pred, y_test)