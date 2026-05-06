import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.mean(np.abs(y_true - y_pred) / np.maximum(denom, 1e-8))*100

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

import numpy as np

def money(y_true, y_pred, price, selling_price, buying_price):
    """
    Works with scalars or numpy arrays.

    y_true - true consumption
    y_pred - predicted consumption
    price - day ahead price
    selling_price - price for selling excess
    buying_price - price for buying deficit
    """

    # convert everything to arrays (scalars will still work)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    price = np.asarray(price/1000, dtype=float)
    selling_price = np.asarray(selling_price/1000, dtype=float)
    buying_price = np.asarray(buying_price/1000, dtype=float)

    min_spend = y_true * price
    ordered_amount = y_pred * price
    diff = y_pred - y_true

    # element-wise logic
    real_spend = (
        ordered_amount
        - np.where(diff > 0, selling_price * diff, 0.0)
        - np.where(diff < 0, buying_price * diff, 0.0)
    )

    return np.sum(real_spend - min_spend)

def money_pct(y_true, y_pred, price, selling_price, buying_price):
    """
    Works with scalars or numpy arrays.

    y_true - true consumption
    y_pred - predicted consumption
    price - day ahead price
    selling_price - price for selling excess
    buying_price - price for buying deficit
    """

    # convert everything to arrays (scalars will still work)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    price = np.asarray(price/1000, dtype=float)
    selling_price = np.asarray(selling_price/1000, dtype=float)
    buying_price = np.asarray(buying_price/1000, dtype=float)

    min_spend = y_true * price
    ordered_amount = y_pred * price
    diff = y_pred - y_true

    # element-wise logic
    real_spend = (
        ordered_amount
        - np.where(diff > 0, selling_price * diff, 0.0)
        - np.where(diff < 0, buying_price * diff, 0.0)
    )

    return np.sum(real_spend - min_spend)/(np.sum(min_spend)/100)