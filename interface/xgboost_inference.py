import pandas as pd
import time
import xgboost as xgb
import torch
from tqdm.auto import tqdm

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

from inference_data_preprocessing import inference_preprocessing_full

Y_COL        = 'sum_of_kWh'
GROUP_COL    = 'eic_code'
TIME_IDX_COL = "time_idx"
HORIZON      = 48

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
    df["diff_48h"] = (
        grouped.shift(HORIZON) - grouped.shift(HORIZON + 48)
    ).astype("float32")

    df["diff_168h"] = (
        grouped.shift(HORIZON) - grouped.shift(HORIZON + 168)
    ).astype("float32")

    return df

def undo_ohe(df, col_name):
    # find all one-hot columns for this feature
    ohe_cols = [c for c in df.columns if c.startswith(col_name + "_")]

    # recover original category
    df[col_name] = df[ohe_cols].idxmax(axis=1).str.replace(col_name + "_", "", regex=False)

    # drop one-hot columns
    df = df.drop(columns=ohe_cols)

    return df

def predict_xgboost(df, start_dt, end_dt):

    print("Loading data…")
    df = inference_preprocessing_full(df, start_dt, end_dt, ohe=True)

    season_map = {
        12: "1", 1: "1", 2: "1",
        3: "2",  4: "2", 5: "2",
        6: "3",  7: "3", 8: "3",
        9: "4", 10: "4", 11: "4",
    }
    df["season"] = df["datetime"].dt.month.map(season_map)

    print("Adding lag features …")
    df = add_lag_features(df, GROUP_COL, Y_COL)

    df = pd.get_dummies(
        df,
        columns=[GROUP_COL]
    )

    print(f"Shape after feature engineering: {df.shape}")

    loaded_model = xgb.XGBRegressor()
    loaded_model.load_model("models/xgb_model.json")

    predict_df = df[df[Y_COL].isna()].copy()

    predict_X = predict_df.drop([Y_COL, 'datetime'], axis=1)

    print(predict_X[["latitude", "longitude"]])

    predict_X["latitude"] = predict_X["latitude"].astype("float32")
    predict_X["longitude"] = predict_X["longitude"].astype("float32")

    missing_cols = set(loaded_model.feature_names_in_) - set(predict_X.columns)
    extra_cols = set(predict_X.columns) - set(loaded_model.feature_names_in_)

    for c in missing_cols:
        predict_X[c] = 0

    predict_X = predict_X[loaded_model.feature_names_in_]

    y_preds = loaded_model.predict(predict_X).clip(min=0)

    predict_df[Y_COL] = y_preds

    predict_df = undo_ohe(predict_df, GROUP_COL)

    return predict_df[[GROUP_COL, 'datetime', Y_COL,
                       'dam_price', 'buy_bm_price', 'sell_bm_price'
                       ]]


# testing
if __name__ == "__main__":
    df = pd.read_excel("../data/inference/one_station_sample.xlsx")

    df['Дата'] = pd.to_datetime(df['Дата'])

    df_preds = predict_xgboost(df,
    start_dt="2025-09-01",
    end_dt="2025-10-30")

    print(df_preds)
    print(df_preds[Y_COL].describe())