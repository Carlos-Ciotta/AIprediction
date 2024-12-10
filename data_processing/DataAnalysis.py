import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

def OLS(x, y):
    independent_sm = sm.add_constant(x)
    model = sm.OLS(y, independent_sm)
    results = model.fit()

    return results.summary()

def VIF(x):
    vif = pd.DataFrame()
    vif["VIF"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    vif["features"] = x.columns
    return vif

def shapiro(x):
    stat, p = shapiro(x)
    return str('Statistics=%.3f, p=%f' % (stat, p))

def plot_correlation(df, features):
    var_ind = df[features]

    corr = var_ind.corr(method = 'pearson').abs()

    plt.figure(figsize=(8, 6)) 
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')

    return plt.show()

__all__ = [plot_correlation, VIF, shapiro, OLS]