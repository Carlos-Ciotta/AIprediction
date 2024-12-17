from methods.GRUNetwork import GRUNetwork
import data_processing.DatasetProcessing as dp
import data_processing.NumpyProcessingDL as tnpdl
import os
import pandas as pd
from methods.Regressors import lasso_regression
from methods.Regressors import ridge_regression
from methods.Regressors import random_forest
import matplotlib.pyplot as plt  # For creating visualizations
import numpy as np

features_consumption = ['solarradiation','humidity','holiday'
                        ,'month','hour','Clear','Overcast','Partially cloudy',"Rain, Partially cloudy",
                        'Domingo','Quarta','Quinta','Segunda','Sexta','Sábado','Terça'
                        ,'Portugal','Active Energy (MWh) - Porto']
                        
features_price = ['solarradiation','Hídrica','Eólica','Solar',
'Gás Natural - Ciclo Combinado','Gás natural - Cogeração','Importação',
'Bombagem','Portugal'
,'Active Energy (MWh) - Porto']

consumption_pred = ['solarradiation','humidity','holiday'
                        ,'month','hour','Clear','Overcast','Partially cloudy',"Rain, Partially cloudy",
                        'Domingo','Quarta','Quinta','Segunda','Sexta','Sábado','Terça'
                        ,'Portugal']

price_pred = ['solarradiation','Hídrica','Eólica','Solar',
'Gás Natural - Ciclo Combinado','Gás natural - Cogeração','Importação',
'Bombagem'
,'Active Energy (MWh) - Porto']

epochs_c = 600
test_rate_c = 0.2
eta_c = 1e-2

epochs_p = 600
test_rate_p = 0.2
eta_p = 1e-2

def rnn_modeling(features, path, test_rate, epochs,eta, out):
#this function creates a rnn model
    #declare params
    batch_size = 32
    #process data to numpy
    x_train,x_test, y_train, y_test, scalerx, scalery = tnpdl.dataProcessing_toNumpy(data[features], test_rate,out)

    #creates rnn GRU
    rnn = GRUNetwork(x_train,eta,epochs)

    #train model
    rnn.train(x_train, y_train, epochs, batch_size, x_test,y_test)

    #save model
    rnn.save_model(path)

    #predict
    y_pred, y_test = rnn.predict(x_test, y_test, scalery)

    #calculate R2 and MSE to future comparation with regression
    mse,r2 = rnn.evaluate(y_pred, y_test)
    
    return y_pred, mse, r2

def load_model(features, path, test_rate, epochs,eta, out):
#this function load a rnn model
    x_train,x_test, y_train, y_test, scalerx, scalery = tnpdl.dataProcessing_toNumpy(data[features], test_rate, out)

    #init model class
    rnn = GRUNetwork(x_train,eta,epochs)

    #load model
    rnn.load_model(path)

    #predict
    y_pred, y_test = rnn.predict(x_test, y_test, scalery)

    #calculate R2 and MSE to future comparation with regression
    mse,r2 = rnn.evaluate(y_pred, y_test)

    return y_pred, mse, r2

def plot_scatterplot(y_pred,type):
    n_repeats = (y_pred.shape[0] // 24) + 1
    hours = np.tile(np.arange(0, 24), n_repeats)[:(y_pred.shape[0])]  #24 hours interval
    # Ploting scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(hours, y_pred, color='b', alpha=0.6)
    if(type == 'consumption'):
        plt.title("Energy Consumption Porto")
        plt.xlabel("Hours")
        plt.ylabel("Energy Consumption")
    elif(type=='price'):
        plt.title("Energy Price Porto")
        plt.xlabel("Hours")
        plt.ylabel("Price")

    return plt.show()
    
if __name__ == "__main__":
    #first verify if the processed dataset unifying every csv exists
    try:
        if(os.path.exists('datasets/data_training.csv')):
            data = pd.read_csv('datasets/data_training.csv')

        #otherwise creates it
        else:
            dp.merged_processing('datasets/posta_code_energy_consumption_porto2023.csv', 
                                 'datasets/weather_porto 2022-08-30 to 2023-09-30.csv',
                                  'datasets/Porto_holidays.txt',
                                  'datasets/prices_energy_20220101_20241209.csv',
                                  'datasets/energy_production_portugal_20220101_20240912.csv')
            data = pd.read_csv('datasets/data_training.csv')
    #in case o error, print it
    except:
        print('Erros')

    #no error -> GRU and regression models existence verification
    else:
        if((os.path.exists('RNNmodels/consumption_model4.h5')) and os.path.exists('RNNmodels/price_model.h5')):
            #load model
            y_predGRUConsumption, mse_GRUConsumption, r2_GRUConsumption = load_model(features_consumption, 'RNNmodels/consumption_model4.h5', test_rate_c, epochs_c,eta_c, 'Active Energy (MWh) - Porto')
            #load model
            y_predGRUPrice, mse_GRUPrice, r2_GRUPrice =load_model(features_price, 'RNNmodels/price_model.h5', test_rate_p, epochs_p,eta_p, 'Portugal')

        else:
            if(not (os.path.exists('RNNmodels/price_model.h5'))):
                y_predGRUPrice, mse_GRUPrice, r2_GRUPrice = rnn_modeling(features_price, 'price_model.h5', test_rate_p, epochs_p,eta_p,'Portugal')

            if(not (os.path.exists('RNNmodels/consumption_model4.h5'))):
                y_predGRUConsumption, mse_GRUConsumption, r2_GRUConsumption = rnn_modeling(features_consumption, 'consumption_model4.h5', test_rate_c, epochs_c,eta_c, 'Active Energy (MWh) - Porto')
        
        #regressor
        impPrice, y_pred_forestPrice, mse_forestPrice, r2_forestPrice = random_forest(data[price_pred], data['Portugal'], test_rate_p, price_pred)
        y_pred_lassoPrice, mse_lassoPrice, r2_lassoPrice = lasso_regression(data[price_pred],data['Portugal'],test_rate_p)
        y_pred_ridgePrice, mse_ridgePrice, r2_ridgePrice = ridge_regression(data[price_pred],data['Portugal'],test_rate_p)
            
        #regressor
        impConsumption, y_pred_forestConsumption, mse_forestConsumption, r2_forestConsumption = random_forest(data[consumption_pred], data['Active Energy (MWh) - Porto'], test_rate_c, consumption_pred)
        y_pred_lassoConsumption, mse_lassoConsumption, r2_lassoConsumption = lasso_regression(data[consumption_pred],data['Active Energy (MWh) - Porto'],test_rate_c)
        y_pred_ridgeConsumption, mse_ridgeConsumption, r2_ridgeConsumption = ridge_regression(data[consumption_pred],data['Active Energy (MWh) - Porto'],test_rate_c)

        measurements = pd.DataFrame({'Method (PRICE)':['Ridge', 'Lasso', 'Random Forest', 'GRU'],
                                      'R2 - Price': [r2_ridgePrice, r2_lassoPrice, r2_forestPrice, r2_GRUPrice],
                                      'MSE - Price': [mse_ridgePrice, mse_lassoPrice, mse_forestPrice, mse_GRUPrice],

                                      'Method (CONSUMPTION)':['Ridge', 'Lasso', 'Random Forest', 'GRU'],
                                      'R2 - Consumption': [r2_ridgeConsumption, r2_lassoConsumption, r2_forestConsumption, r2_GRUConsumption],
                                      'MSE - Consumption': [mse_ridgeConsumption, mse_lassoConsumption, mse_forestConsumption, mse_GRUConsumption],
                                      })
        
        print(measurements)

        #selecting the lowest r2
        min_index = measurements['R2 - Price'].idxmax()
        best_method = measurements.loc[min_index, 'Method (PRICE)']

        if(best_method == 'Ridge'):
            plot_scatterplot(y_pred_ridgePrice, 'price')
        elif(best_method == 'Lasso'):
            plot_scatterplot(y_pred_lassoPrice, 'price')
        elif(best_method == 'Random Forest'):
            plot_scatterplot(y_pred_forestPrice, 'price')
        elif(best_method == 'GRU'):
            plot_scatterplot(y_predGRUPrice, 'price')

        #selecting the lowest r2
        min_index = measurements['R2 - Consumption'].idxmin()
        best_method = measurements.loc[min_index, 'Method (CONSUMPTION)']

        if(best_method == 'Ridge'):
            plot_scatterplot(y_pred_ridgeConsumption, 'consumption')
        elif(best_method == 'Lasso'):
            plot_scatterplot(y_pred_lassoConsumption, 'consumption')
        elif(best_method == 'Random Forest'):
            plot_scatterplot(y_pred_forestConsumption, 'consumption')
        elif(best_method == 'GRU'):
            plot_scatterplot(y_predGRUConsumption, 'consumption')