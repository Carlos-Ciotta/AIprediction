from data_processing.NumpyProcessingR import dataProcessing_toNumpy
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def lasso_regression(x, y, test_rate):
    x_train, x_test, y_train, y_test, scalerx, scalery = dataProcessing_toNumpy(x, y,test_rate)
    lasso = Lasso(alpha=1e-2, max_iter=2000)

    # Treinamento do modelo
    lasso.fit(x_train, y_train)

    # Previsões
    y_pred = lasso.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return y_pred, mse, r2

def ridge_regression(x, y, test_rate):
    x_train, x_test, y_train, y_test, scalerx, scalery = dataProcessing_toNumpy(x, y,test_rate)
    ridge = Ridge(alpha=1e-2,max_iter=2000, solver='sag')

    # Treinamento do modelo
    ridge.fit(x_train, y_train)

    # Previsões
    y_pred = ridge.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return y_pred, mse, r2

def random_forest(x,y, test_rate, columns):
    import pandas as pd
    model = RandomForestRegressor()
    x_train, x_test, y_train, y_test, scalerx, scalery = dataProcessing_toNumpy(x, y,test_rate)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)

    importances = model.feature_importances_
    feature_names = columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return importance_df, y_pred, mse, r2

def evaluate (y_test, y_pred):
    import numpy as np
    #mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    #rmse = np.sqrt(mse)
    #r2 = r2_score(y_test, y_pred)

    #print("Desempenho do Modelo:")
    #print(f"MAE: {mae:.2f}")
    #print(f"MSE: {mse:.2f}")
    #print(f"RMSE: {rmse:.2f}")
    #print(f"R²: {r2:.2f}")

    return mse
__all__=[ridge_regression, lasso_regression]