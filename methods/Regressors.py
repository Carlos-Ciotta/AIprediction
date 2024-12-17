from data_processing.NumpyProcessingR import dataProcessing_toNumpy
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def lasso_regression(x, y, test_rate):
    # Function to train a Lasso regression model.
    x_train, x_test, y_train, y_test, scalerx, scaler_y = dataProcessing_toNumpy(x, y, test_rate)
    lasso = Lasso(alpha=1e-2, max_iter=2000)
    # Initializing the Lasso regression model with a regularization strength of 0.01 and maximum iterations of 2000.
    lasso.fit(x_train, y_train)
    y_pred = lasso.predict(x_test)
    # Inverses the scaling of predictions and ground truth
    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
    y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    mse = mean_squared_error(y_test, y_pred)
    # Calculating the Mean Squared Error (MSE) to evaluate the model's prediction error.
    r2 = r2_score(y_test, y_pred)
    # Calculating the R-squared (R²) to measure the goodness of fit of the model.
    
    return y_pred, mse, r2

def ridge_regression(x, y, test_rate):
    # Function to train a Ridge regression model.
    x_train, x_test, y_train, y_test, scalerx, scaler_y = dataProcessing_toNumpy(x, y, test_rate)
    ridge = Ridge(alpha=1e-2, max_iter=2000, solver='sag')
    # Initializing the Ridge regression model with a regularization strength of 0.01, 
    # maximum iterations of 2000, and using the 'sag' solver for optimization.
    ridge.fit(x_train, y_train)
    y_pred = ridge.predict(x_test)
    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
    y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    mse = mean_squared_error(y_test, y_pred)
    # Calculating the Mean Squared Error (MSE) to evaluate the model's prediction error.

    r2 = r2_score(y_test, y_pred)
    # Calculating the R-squared (R²) to measure the goodness of fit of the model.

    return y_pred, mse, r2

def random_forest(x, y, test_rate, columns):
    # Function to train a Random Forest Regressor model and evaluate feature importance.
    import pandas as pd
    model = RandomForestRegressor()
    x_train, x_test, y_train, y_test, scalerx, scaler_y = dataProcessing_toNumpy(x, y, test_rate)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
    y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    importances = model.feature_importances_
    # Extracting the feature importance values from the trained Random Forest model.

    feature_names = columns
    # Assigning column names to the features for better interpretability.

    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    # Sorting the feature importance values in descending order for better visualization.

    mse = mean_squared_error(y_test, y_pred)
    # Calculating the Mean Squared Error (MSE) to evaluate the model's prediction error.

    r2 = r2_score(y_test, y_pred)
    # Calculating the R-squared (R²) to measure the goodness of fit of the model.

    return importance_df, y_pred, mse, r2

__all__ = [ridge_regression, lasso_regression]