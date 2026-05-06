#%%
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
#%%
def smallest_int_dtype(min_val: int, max_val: int, signed: bool = True) -> str:
    if signed:
        if np.iinfo(np.int8).min <= min_val <= max_val <= np.iinfo(np.int8).max:
            return "int8"
        if np.iinfo(np.int16).min <= min_val <= max_val <= np.iinfo(np.int16).max:
            return "int16"
        if np.iinfo(np.int32).min <= min_val <= max_val <= np.iinfo(np.int32).max:
            return "int32"
        return "int64"
    else:
        if 0 <= min_val <= max_val <= np.iinfo(np.uint8).max:
            return "uint8"
        if 0 <= min_val <= max_val <= np.iinfo(np.uint16).max:
            return "uint16"
        if 0 <= min_val <= max_val <= np.iinfo(np.uint32).max:
            return "uint32"
        return "uint64"


def optimize_df_for_memory(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Converts columns to smaller dtypes.
    For low-decimal float columns, stores scaled integers if that beats float32.
    Returns:
        optimized_df
        metadata dict with scaling info
    """
    meta = {}

    for col in df.columns:
        s = df[col]

        # bool-like columns
        unique_non_null = set(s.dropna().unique())
        if unique_non_null.issubset({0, 1, True, False}):
            if col == "Група":
                df[col] = s.astype("bool")
                meta[col] = {"stored_as": "bool", "scale": 1}
                continue

        # integer columns
        if pd.api.types.is_integer_dtype(s):
            mn, mx = int(s.min()), int(s.max())
            dtype = smallest_int_dtype(mn, mx, signed=(mn < 0))
            df[col] = s.astype(dtype)
            meta[col] = {"stored_as": dtype, "scale": 1}
            continue

        # float columns
        if pd.api.types.is_float_dtype(s):
            # estimate visible decimal precision
            non_null = s.dropna()
            if len(non_null) == 0:
                df[col] = s.astype("float32")
                meta[col] = {"stored_as": "float32", "scale": 1}
                continue

            decimals = non_null.astype(str).apply(
                lambda x: len(x.split(".")[1].rstrip("0")) if "." in x else 0
            ).max()

            # try scaled integer
            if decimals <= 3:
                scale = 10 ** decimals
                scaled = np.round(s * scale)

                mn = int(np.nanmin(scaled))
                mx = int(np.nanmax(scaled))
                int_dtype = smallest_int_dtype(mn, mx, signed=(mn < 0))

                int_bytes = np.dtype(int_dtype).itemsize
                float32_bytes = np.dtype("float32").itemsize

                if int_bytes < float32_bytes:
                    df[col] = scaled.astype(int_dtype)
                    meta[col] = {"stored_as": int_dtype, "scale": scale}
                else:
                    df[col] = s.astype("float32")
                    meta[col] = {"stored_as": "float32", "scale": 1}
            else:
                df[col] = s.astype("float32")
                meta[col] = {"stored_as": "float32", "scale": 1}

    return df, meta
#%%
train = pd.read_parquet("../data/money_calc/train.parquet")

train = train.reset_index(drop=True)

train, train_meta = optimize_df_for_memory(train)

val = pd.read_parquet("../data/money_calc/val.parquet")

val = val.reset_index(drop=True)

val, val_meta = optimize_df_for_memory(val)
#%%
y_col = 'Money_spent'
#%%
def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.mean(np.abs(y_true - y_pred) / np.maximum(denom, 1e-8))

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
#%%
# model training script here
#%%
test = pd.read_parquet("../data/money_calc/test.parquet")

test = test.reset_index(drop=True)

test, test_meta = optimize_df_for_memory(test)
#%%
# inference script here
#%%
print("\nValidation metrics")
print("SMAPE:", smape(val[y_col], val_pred))
print("RMSE :", rmse(val[y_col], val_pred))
print("MAPE :", mape(val[y_col], val_pred))

print("\nTest metrics")
print("SMAPE:", smape(test[y_col], test_pred))
print("RMSE :", rmse(test[y_col], test_pred))
print("MAPE :", mape(test[y_col], test_pred))