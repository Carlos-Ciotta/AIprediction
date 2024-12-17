# Importing necessary libraries
import matplotlib.pyplot as plt  # For creating visualizations
import seaborn as sns  # For advanced statistical data visualizations
import statsmodels.api as sm  # For statistical modeling and analysis
from scipy.stats import shapiro  # For performing the Shapiro-Wilk normality test
from statsmodels.stats.outliers_influence import variance_inflation_factor  # For calculating Variance Inflation Factor (VIF)
import pandas as pd  # For handling data in tabular format

# Function to perform Ordinary Least Squares (OLS) regression
def OLS(x, y):
    # Add a constant (intercept) term to the independent variables
    independent_sm = sm.add_constant(x)

    # Create an OLS regression model with the provided independent and dependent variables
    model = sm.OLS(y, independent_sm)

    results = model.fit()

    return results.summary()

# Function to calculate Variance Inflation Factor (VIF) for independent variables
def VIF(x):
    # Create a DataFrame to store VIF values and their corresponding feature names
    vif = pd.DataFrame()

    # Calculate VIF for each feature in the dataset
    vif["VIF"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    vif["features"] = x.columns

    return vif

# Function to perform the Shapiro-Wilk test for normality
def shapiro(x):
    # Perform the Shapiro-Wilk test on the input data
    stat, p = shapiro(x)

    return str('Statistics=%.3f, p=%f' % (stat, p))

# Function to create a correlation heatmap for the specified features in a DataFrame
def plot_correlation(df, features):
    # Extract the independent variables (features) from the DataFrame
    var_ind = df[features]

    # Compute the Pearson correlation matrix (absolute values)
    corr = var_ind.corr(method='pearson').abs()

    # Create a heatmap to visualize the correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')

    return plt.show()

# Function to create a scatterplot for two variables in a dataset
def scatterplot(data, x, y):
    return sns.scatterplot(data, x, y)

# Specify which functions are part of the public API for this script/module
__all__ = [plot_correlation, VIF, shapiro, OLS, scatterplot]