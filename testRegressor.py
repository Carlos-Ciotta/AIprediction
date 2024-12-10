import data_processing.DatasetProcessing as dp
import os
import pandas as pd
from methods.Regressors import lasso_regression
from methods.Regressors import ridge_regression
from methods.Regressors import random_forest

#'Active Energy (MWh) - Porto'

data = pd.read_csv('datasets/data_training.csv')

consumption_pred = ['solarradiation','humidity','holiday'
                        ,'month','hour','Clear','Overcast','Partially cloudy',"Rain, Partially cloudy",
                        'Domingo','Quarta','Quinta','Segunda','Sexta','Sábado','Terça'
                        ,'Portugal']

price_pred = ['solarradiation','Hídrica','Eólica','Solar',
'Gás Natural - Ciclo Combinado','Gás natural - Cogeração','Importação',
'Bombagem'
,'Active Energy (MWh) - Porto']

x = data[consumption_pred]
y = data['Active Energy (MWh) - Porto']

test_rate = 0.2

if __name__ == "__main__":
        if(os.path.exists('datasets/data_training.csv')):
            imp, y_pred_forest, mse_forest, r2f = random_forest(x, y, test_rate, consumption_pred)
            y_pred_lasso, mse_lasso, r2l = lasso_regression(x,y,test_rate)
            y_pred_ridge, mse_ridge, r2r = ridge_regression(x,y,test_rate)
            
            print(f'\nMSE DOS MODELOS\nLASSO = {mse_lasso}\n\nRIDGE = {mse_ridge}\n\nRANDOM FOREST = {mse_forest}')
            print(f'\nR2 DOS MODELOS\nLASSO = {r2l}\n\nRIDGE = {r2r}\n\nRANDOM FOREST = {r2f}')

        else:
            dp.merged_processing('datasets/posta_code_energy_consumption_porto2023.csv', 
                                 'datasets/weather_porto 2023-01-01 to 2023-09-30.csv',
                                  'datasets/Porto_holidays.txt',
                                  'datasets/prices_energy_20230101_20240101.csv',
                                  'datasets/energy_production_portugal_20230101_20240101.csv')
            data = pd.read_csv('datasets/data_training.csv')

            imp, y_pred_forest = random_forest(x, y, test_rate, consumption_pred)
            y_pred_lasso = lasso_regression(x,y,test_rate)
            y_pred_ridge = ridge_regression(x,y,test_rate)
            