from scipy.stats import pearsonr, spearmanr
import numpy as np


def lcc(y, y_pred):
    """Wrapping the pearsonr function to use it with GridSearchCV"""
    corr = pearsonr(y_pred, y)[0]
    if np.isnan(corr):
        corr = 0.0
    return corr


def srocc(y, y_pred):
    """Wrapping the spearmanr function to use it with GridSearchCV"""
    corr = spearmanr(y_pred, y)[0]
    if np.isnan(corr):
        corr = 0.0
    return corr
