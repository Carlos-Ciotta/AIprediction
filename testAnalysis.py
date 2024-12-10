from data_processing.DatasetProcessing import merged_processing
import os
import pandas as pd
from data_processing.DataAnalysis import plot_correlation, OLS, VIF

consumption_pred = ['solarradiation','humidity','holiday'
                        ,'month','hour','Clear','Overcast','Partially cloudy',"Rain, Partially cloudy",
                        'Domingo','Quarta','Quinta','Segunda','Sexta','Sábado','Terça'
                        ,'Portugal','Active Energy (MWh) - Porto']

price_pred = ['solarradiation','Hídrica','Eólica','Solar',
'Gás Natural - Ciclo Combinado','Gás natural - Cogeração','Importação',
'Bombagem'
,'Active Energy (MWh) - Porto', 'Portugal']

if __name__ == "__main__":
        if(os.path.exists('datasets/data_training.csv')):
            data = pd.read_csv('datasets/data_training.csv')
            data = data[consumption_pred]
            features = data.columns
            plot_correlation(data, features)
            #sum = OLS(data[features], data['Active Energy (MWh) - Porto'])
            vif = VIF(data[features])
            #print(sum)
            print(vif)
            

        else:
            merged_processing('datasets/posta_code_energy_consumption_porto2023.csv', 
                                 'datasets/weather_porto 2022-08-30 to 2023-09-30.csv',
                                  'datasets/Porto_holidays.txt',
                                  'datasets/prices_energy_20220101_20241209.csv',
                                  'datasets/energy_production_portugal_20220101_20240912.csv')
            data = pd.read_csv('datasets/data_training.csv')
            data = data[consumption_pred]
            features = data.columns
            plot_correlation(data, features)
            #sum = OLS(data[features], data['Active Energy (MWh) - Porto'])
            vif = VIF(data[features])
            #print(sum)
            print(vif)
            