import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.mean(np.abs(y_true - y_pred) / np.maximum(denom, 1e-8))

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def money(y_true, y_pred, price, selling_price, buying_price):
    """
    Compute the extra cost (vs perfect forecast) for each sample.

    Over-prediction (y_pred > y_true): excess energy sold back at selling_price.
    Under-prediction (y_pred < y_true): deficit bought at buying_price.

    Returns an array of per-sample extra costs (negative = savings, positive = penalty).
    Works element-wise on scalars or numpy arrays.
    """
    y_true        = np.asarray(y_true,        dtype=float)
    y_pred        = np.asarray(y_pred,        dtype=float)
    price         = np.asarray(price,         dtype=float)
    selling_price = np.asarray(selling_price, dtype=float)
    buying_price  = np.asarray(buying_price,  dtype=float)

    min_spend = y_true * price
    diff      = y_pred - y_true

    selling_additional = np.where(diff > 0, selling_price * (y_true - y_pred), 0.0)
    buying_additional  = np.where(diff < 0, buying_price  * diff,              0.0)

    real_spend = min_spend + selling_additional + buying_additional
    return real_spend - min_spend