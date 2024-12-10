import pandas as pd

def holiday_processing(path):
    dfH = pd.read_csv(path, sep = ';')
    dfH_filtered = dfH.copy()

    ##### CLEAN NaN #####
    dfH_filtered = dfH_filtered.dropna()
    #####################
    return dfH_filtered

def weather_processing(path):
    dfW= pd.read_csv(path, sep=',') #Porto's weather data
    dfW_filtered = dfW[['datetime', 'conditions', 'solarradiation', 'humidity']].copy()

    ##### FILTERING DATE TIME #####
    dfW_filtered['datetime'] = pd.to_datetime(dfW_filtered['datetime'])
    dfW_filtered['datetime'] = pd.to_datetime(dfW_filtered['datetime'], format = '%Y-%m-%d %H:%M:%S')
    dfW_filtered['day'] = dfW_filtered['datetime'].dt.day
    dfW_filtered['month'] = dfW_filtered['datetime'].dt.month
    dfW_filtered['year'] = dfW_filtered['datetime'].dt.year
    dfW_filtered['hour'] = dfW_filtered['datetime'].dt.hour
    dfW_filtered['minute'] = dfW_filtered['datetime'].dt.minute

    dfW_filtered = dfW_filtered.drop(['datetime'], axis = 1)
    dfW_filtered = dfW_filtered.reset_index(drop = True)
    ###############################

    ##### APPLYING ONE HOT ENCODE TO CONDITIONS #####
    conditions_oh = pd.get_dummies(dfW_filtered['conditions'])
    dfW_filtered = dfW_filtered.drop(['conditions'], axis = 1)
    dfW_filtered = dfW_filtered.join(conditions_oh)
    dfW_filtered = dfW_filtered.reset_index(drop = True)
    #################################################

    ##### APPLYING BOOLEAN ON CONDITIONS_OH #####
    df_boolean = dfW_filtered.select_dtypes(include=['bool'])
    dfW_filtered[df_boolean.columns] = df_boolean.astype(int)
    #############################################

    ##### CLEAN NaN #####
    dfW_filtered = dfW_filtered.dropna()
    #####################
    return dfW_filtered

def consumption_processing(path):
    dfC = pd.read_csv(path, sep = ';') #Porto's energy consumption
    dfC_filtered = dfC[['Data/Hora', 'Energia ativa (kWh)', 'Dia da Semana']].copy()

    ##### APPLYING DATETIME #####
    dfC_filtered = dfC_filtered.sort_values('Data/Hora').reset_index(drop=True)
    dfC_filtered['Data/Hora'] = pd.to_datetime(dfC_filtered['Data/Hora'], utc=True)
    dfC_filtered['Data/Hora'] = pd.to_datetime(dfC_filtered['Data/Hora'], format='%Y-%m-%d %H:%M:%S', utc=True)
    dfC_filtered['day'] = dfC_filtered['Data/Hora'].dt.day
    dfC_filtered['month'] = dfC_filtered['Data/Hora'].dt.month
    dfC_filtered['year'] = dfC_filtered['Data/Hora'].dt.year
    dfC_filtered['hour'] = dfC_filtered['Data/Hora'].dt.hour
    dfC_filtered['minute'] = dfC_filtered['Data/Hora'].dt.minute

    dfC_filtered = dfC_filtered.drop(['Data/Hora'], axis = 1)
    dfC_filtered = dfC_filtered.reset_index(drop = True)
    #############################

    ##### SUMMING ACTIVE ENERGY FROM THE SAME HOUR/DAY/MONTH #####
    dfC_filtered = dfC_filtered.groupby(['Dia da Semana', 'day', 'month', 'hour', 'minute', 'year'], as_index=False)['Energia ativa (kWh)'].sum()
    dfC_filtered['Active Energy (MWh) - Porto'] = dfC_filtered['Energia ativa (kWh)'] / 1000
    ##############################################################

    ##### APPLYING ONE HOT TO DIA DA SEMANA (WEEKDAY) #####
    weekday_oh = pd.get_dummies(dfC_filtered['Dia da Semana'])
    dfC_filtered = dfC_filtered.drop(['Dia da Semana'], axis = 1)
    dfC_filtered = dfC_filtered.join(weekday_oh)
    dfC_filtered = dfC_filtered.reset_index(drop = True)
    #######################################################

    ##### APPLYING BOOLEAN ON CONDITIONS_OH #####
    df_boolean = dfC_filtered.select_dtypes(include=['bool'])
    dfC_filtered[df_boolean.columns] = df_boolean.astype(int)
    #############################################

    ##### CLEAN NaN #####
    dfC_filtered = dfC_filtered.dropna()
    #####################
    return dfC_filtered
def generation_processing(path):
    dfG = pd.read_csv(path, sep = ';', skiprows=2) #Portugal's energy Generation
    dfG_filtered = dfG.copy()
    ##### APPLYING DATETIME #####
    dfG_filtered['Data e Hora'] = pd.to_datetime(dfG_filtered['Data e Hora'])
    dfG_filtered['Data e Hora'] = pd.to_datetime(dfG_filtered['Data e Hora'], format='%Y-%m-%d %H:%M:%S')
    dfG_filtered['day'] = dfG_filtered['Data e Hora'].dt.day
    dfG_filtered['month'] = dfG_filtered['Data e Hora'].dt.month
    dfG_filtered['year'] = dfG_filtered['Data e Hora'].dt.year
    dfG_filtered['hour'] = dfG_filtered['Data e Hora'].dt.hour
    dfG_filtered['minute'] = dfG_filtered['Data e Hora'].dt.minute

    dfG_filtered = dfG_filtered.drop(['Data e Hora'], axis = 1)
    dfG_filtered = dfG_filtered.reset_index(drop = True)
    #############################

    ##### SUMMING ACTIVE ENERGY FROM THE SAME HOUR/DAY/MONTH #####
    dfG_filtered = dfG_filtered.groupby(['day', 'month', 'hour', 'year'], as_index=False)[
        ['Hídrica', 'Eólica', 'Solar', 'Biomassa', 'Ondas',
        'Gás Natural - Ciclo Combinado', 'Gás natural - Cogeração', 'Carvão',
        'Outra Térmica', 'Importação', 'Exportação', 'Bombagem',
        'Injeção de Baterias', 'Consumo Baterias', 'Consumo']].sum()
    ##############################################################

    ##### CLEAN NaN #####
    dfG_filtered = dfG_filtered.dropna()
    #####################
    return dfG_filtered

def price_processing(path):
    dfP = pd.read_csv(path, sep=';', skiprows=2)
    dfP_filtered = dfP[['Data', 'Hora', 'Portugal']].copy()

    ##### CLEAN NaN AND ZEROS #####
    dfP_filtered = dfP_filtered.dropna()
    dfP_filtered = dfP_filtered[(dfP_filtered['Portugal'] >0)]
    #####################
    
    ##### APPLYING DATETIME #####
    dfP_filtered['Data'] = pd.to_datetime(dfP_filtered['Data'])
    dfP_filtered['Data'] = pd.to_datetime(dfP_filtered['Data'], format='%Y-%m-%d')
    dfP_filtered['day'] = dfP_filtered['Data'].dt.day
    dfP_filtered['month'] = dfP_filtered['Data'].dt.month
    dfP_filtered['year'] = dfP_filtered['Data'].dt.year
    #############################
    
    ##### RENAMING HOURS FOR FUTURE MERGE #####
    dfP_filtered = dfP_filtered.rename(columns={'Hora': 'hour'})
    ###########################################
    
    return dfP_filtered

def merged_processing(path_dfC,path_dfW,path_dfH,path_dfP,path_dfG):
    dfP = price_processing(path_dfP)
    dfW = weather_processing(path_dfW)
    dfG = generation_processing(path_dfG)
    dfH = holiday_processing(path_dfH)
    dfC = consumption_processing(path_dfC)

    ##### JOIN HOLIDAYS AS BOOLEAN #####
    dfMerged = pd.merge(dfW, dfH, on=['day', 'month', 'year'], how='left', indicator=True)
    dfMerged['holiday'] = dfMerged['_merge'].apply(lambda x: 1 if x == 'both' else 0)
    dfMerged = dfMerged.drop(columns=['_merge'])
    dfMerged = dfMerged.reset_index(drop=True)
    ####################################

    ##### MERGE WITH CONSUMPTION #####
    dfMerged = pd.merge(dfMerged, dfC, on=['day', 'month', 'year', 'hour'], how='inner')
    ##################################

    ##### MERGE WITH GENERATION #####
    dfMerged = pd.merge(dfMerged, dfG, on=['day', 'month', 'year', 'hour'], how='inner')
    dfMerged['Total Consumption (Portugal)']=dfMerged[['Consumo', 'Consumo Baterias']].sum(axis=1)
    constraint = (dfMerged['Active Energy (MWh) - Porto'].sum())/(dfMerged['Total Consumption (Portugal)'].sum())

    dfMerged[['Hídrica', 'Eólica', 'Solar', 'Biomassa', 'Ondas', 
        'Gás Natural - Ciclo Combinado', 'Gás natural - Cogeração', 
        'Carvão', 'Outra Térmica', 'Importação', 'Exportação', 
        'Bombagem', 'Injeção de Baterias']] *= constraint
    #################################

    ##### MERGE WITH PRICE #####
    dfMerged = pd.merge(dfMerged, dfP, on=['day', 'month', 'year', 'hour'], how='inner')
    dfMerged = dfMerged.drop(columns = ['Data'])
    dfMerged = dfMerged.reset_index(drop=True)
    ############################
    
    ##### JOIN OUTPUT COLUMN IN LAST ROW #####
    col_to_move = dfMerged.pop('Active Energy (MWh) - Porto')  #Remove Column
    dfMerged['Active Energy (MWh) - Porto'] = col_to_move  # Add column
    ##########################################

    ##### CLEAN NaN #####
    dfMerged = dfMerged.dropna()
    #####################

    ##### CLEAN SOME COLUMNS #####
    deletable = ['minute_x','minute_y','Energia ativa (kWh)', 'day', 'year']
    dfMerged = dfMerged.loc[:, ~dfMerged.columns.duplicated()]
    dfMerged = dfMerged.drop(deletable, axis = 1)
    dfMerged = dfMerged.reset_index(drop=True)
    ##############################
    return dfMerged.to_csv('datasets/data_training.csv', index=False)

__all__ = [merged_processing]