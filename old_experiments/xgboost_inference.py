import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
import torch
from tqdm.auto import tqdm
import gc

if torch.cuda.is_available():
    try:
        # Quick smoke-test: train a tiny model with device='cuda'
        _probe = xgb.XGBRegressor(n_estimators=1, device='cuda')
        _probe.fit([[0], [1]], [0, 1])
        DEVICE = 'cuda'
    except Exception:
        DEVICE = 'gpu'   # fall back to OpenCL build
else:
    DEVICE = 'cpu'

# With GPU training XGboost manages its own parallelism;
# n_jobs > 1 on GPU gives no benefit and can cause contention.
N_JOBS = 1 if DEVICE != 'cpu' else -1

print(f"XGboost  : {xgb.__version__}")
print(f"CUDA avail: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU       : {torch.cuda.get_device_name(0)}")
print(f"XGboost device → '{DEVICE}'")

HORIZON = 48
# Lags to create (all >= HORIZON to be safe for recursive prediction)
LAG_HOURS = [
    HORIZON,
    HORIZON + 1,
    HORIZON + 2,
    HORIZON + 3,
    HORIZON + 4,
    24 * 2,   # 48 h  (same as HORIZON)
    24 * 3,
    24 * 4,
    24 * 5,
    24 * 6,
    24 * 7,   # 168 h  – same hour, one week ago
    24 * 14,  # two weeks
]
LAG_HOURS = sorted(set(LAG_HOURS))

# Rolling windows applied at offset=HORIZON (shift first, then roll)
ROLL_WINDOWS = [24, 48, 168]

def add_lag_features(df: pd.DataFrame, group_col: str, target_col: str) -> pd.DataFrame:
    """
    Efficiently add lag, rolling, and diff features grouped by `group_col`.
    Operates in-place and returns the modified DataFrame.
    """

    # --- sort once ---
    df = df.sort_values([group_col, "time_idx"]).reset_index(drop=True)

    # --- grouped target ---
    grouped = df.groupby(group_col)[target_col]

    # --- lag features ---
    for lag in tqdm(LAG_HOURS, desc="Creating lag features"):
        df[f"lag_{lag}h"] = grouped.shift(lag).astype("float32")

    # --- shifted target (for leakage-safe rolling) ---
    shifted = grouped.shift(HORIZON)

    # IMPORTANT: group only once
    grouped_shifted = shifted.groupby(df[group_col])

    # --- rolling features ---
    for w in tqdm(ROLL_WINDOWS, desc="Creating rolling features"):
        roll = grouped_shifted.rolling(
            window=w,
            min_periods=max(1, w // 4)
        )

        # compute both stats from same rolling object
        df[f"roll_mean_{w}h"] = (
            roll.mean()
            .reset_index(level=0, drop=True)
            .astype("float32")
        )

        df[f"roll_std_{w}h"] = (
            roll.std()
            .reset_index(level=0, drop=True)
            .astype("float32")
        )

    # --- diff features ---
    # reuse grouped (fast, vectorized)
    df["diff_24h"] = (
        grouped.shift(HORIZON) - grouped.shift(HORIZON + 24)
    ).astype("float32")

    df["diff_168h"] = (
        grouped.shift(HORIZON) - grouped.shift(HORIZON + 168)
    ).astype("float32")

    return df

GROUP_COL = "eic_code"

def run_inference(df):
    print("Adding lag features to all_data …")
    df = add_lag_features(df, "eic_code", "sum_of_kWh")
    print(f"Shape after feature engineering: {df.shape}")
    cols_to_ohe = [
        'dso_desc', 'station_type', 'oblast', GROUP_COL,
        'Month', 'Day', 'Hour', 'day_of_week', 'season'
    ]

    df = pd.get_dummies(
        df,
        columns=cols_to_ohe,
        drop_first=False
    )

    loaded_model = XGBRegressor()
    loaded_model.load_model("models/xgb_model.json")