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
    min_spend = y_true * price

    if y_pred - y_true > 0:
        selling_additional = selling_price * (y_true - y_pred)
    else:
        selling_additional = 0

    if y_pred - y_true < 0:
        buying_additional = buying_price * (y_pred - y_true)
    else:
        buying_additional = 0

    real_spend = min_spend + selling_additional + buying_additional

    return real_spend - min_spend