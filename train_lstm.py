#%%
# ==============================================================================
# Cell 1 – Imports & Setup
# ==============================================================================
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error

# PyTorch Forecasting + Lightning
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer, TorchNormalizer
from pytorch_forecasting.metrics import SMAPE, RMSE, MAE, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

print(f"PyTorch version  : {torch.__version__}")
print(f"CUDA available   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU              : {torch.cuda.get_device_name(0)}")

#%%
# ==============================================================================
# Cell 2 – Memory-Optimization Helpers  (unchanged from original script)
# ==============================================================================
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

        unique_non_null = set(s.dropna().unique())
        if unique_non_null.issubset({0, 1, True, False}):
            if col == "Група":
                df[col] = s.astype("bool")
                meta[col] = {"stored_as": "bool", "scale": 1}
                continue

        if pd.api.types.is_integer_dtype(s):
            mn, mx = int(s.min()), int(s.max())
            dtype = smallest_int_dtype(mn, mx, signed=(mn < 0))
            df[col] = s.astype(dtype)
            meta[col] = {"stored_as": dtype, "scale": 1}
            continue

        if pd.api.types.is_float_dtype(s):
            non_null = s.dropna()
            if len(non_null) == 0:
                df[col] = s.astype("float32")
                meta[col] = {"stored_as": "float32", "scale": 1}
                continue

            decimals = non_null.astype(str).apply(
                lambda x: len(x.split(".")[1].rstrip("0")) if "." in x else 0
            ).max()

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
# ==============================================================================
# Cell 3 – Metric Helpers  (unchanged from original script)
# ==============================================================================
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

#%%
# ==============================================================================
# Cell 4 – Configuration  ← adjust these before running
# ==============================================================================

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAIN_PATH = "data/money_calc/train.parquet"
VAL_PATH   = "data/money_calc/val.parquet"
TEST_PATH  = "data/money_calc/test.parquet"

# ── Target ─────────────────────────────────────────────────────────────────────
Y_COL = "Money_spent"

# ── Station identifier column ──────────────────────────────────────────────────
# PyTorch Forecasting needs a single string column that identifies each time series.
# We will derive it from the one-hot EIC columns below (see Cell 5).
GROUP_COL = "station_id"

# ── Forecasting horizon & encoder length ───────────────────────────────────────
# Predict this many hours ahead
MAX_PREDICTION_LENGTH = 24          # 24 h forecast
MAX_ENCODER_LENGTH    = 7 * 24      # look-back window: 7 days

# ── Training hyper-parameters ──────────────────────────────────────────────────
BATCH_SIZE    = 64
MAX_EPOCHS    = 50
LEARNING_RATE = 3e-3
HIDDEN_SIZE   = 64                  # LSTM hidden units (TFT parameter)
ATTENTION_HEAD_SIZE = 4
DROPOUT       = 0.1
HIDDEN_CONTINUOUS_SIZE = 32

# ── Numerical real-valued covariates known at forecast time (time-varying) ─────
FUTURE_REALS = [
    "temperature_2m", "apparent_temperature", "dew_point_2m",
    "relative_humidity_2m", "precipitation", "rain", "snowfall",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "surface_pressure", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "shortwave_radiation", "diffuse_radiation", "direct_normal_irradiance",
    # calendar — always known in advance
    "Hour", "day_of_week", "season_number", "Month", "Day",
    # electricity prices — assume known schedule
    "Average of Ціна розподілу ЕЕ грн. без ПДВ/кВт*год",
    "Average of Ціна ЕЕ грн. без ПДВ/кВт*год",
]

# ── Numerical real-valued covariates NOT known at forecast time ─────────────────
PAST_REALS = []   # add any columns that are only available historically

# ── Static numerical covariates (one value per station, time-invariant) ─────────
STATIC_REALS = [
    "GPS-координати - Широта",
    "GPS-координати - Довгота",
]

# ── Categorical time-varying known covariates ───────────────────────────────────
# Calendar categoricals — always known in the future
TIME_VARYING_KNOWN_CATS = ["Month_cat", "day_of_week_cat", "season_cat"]  # created in Cell 5

# ── Static categoricals ─────────────────────────────────────────────────────────
# One-hot encoded columns will be collapsed back to a single categorical.
# Any remaining OHE groups (Тип, Область, …) are handled similarly.
STATIC_CATS = ["station_id", "Тип_cat", "Область_cat"]   # created in Cell 5

#%%
# ==============================================================================
# Cell 5 – Load & Pre-process Data
# ==============================================================================

def recover_station_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    The 400 'EIC-код_<station>' columns are one-hot encoded.
    Collapse them into a single station_id string column.
    """
    eic_cols = [c for c in df.columns if c.startswith("EIC-код_")]
    # argmax across OHE columns → station name
    df[GROUP_COL] = df[eic_cols].idxmax(axis=1).str.replace("EIC-код_", "", regex=False)
    df = df.drop(columns=eic_cols)
    return df


def recover_ohe_cat(df: pd.DataFrame, prefix: str, new_col: str) -> pd.DataFrame:
    """
    Collapse a set of OHE columns back to a single categorical column.
    prefix  – common column prefix, e.g. 'Тип_'
    new_col – name for the resulting categorical column
    """
    ohe_cols = [c for c in df.columns if c.startswith(prefix)]
    if not ohe_cols:
        return df
    df[new_col] = df[ohe_cols].idxmax(axis=1).str.replace(prefix, "", regex=False)
    df = df.drop(columns=ohe_cols)
    return df


def build_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    PyTorch Forecasting requires a monotonically increasing integer time_idx
    that is *shared* across all groups (think of it as the global hour counter).
    """
    df["datetime"] = pd.to_datetime(
        df[["Year", "Month", "Day", "Hour"]].rename(
            columns={"Year": "year", "Month": "month", "Day": "day", "Hour": "hour"}
        )
    )
    # Global hourly index starting from the minimum timestamp in the dataset
    min_dt = df["datetime"].min()
    df["time_idx"] = ((df["datetime"] - min_dt).dt.total_seconds() / 3600).astype(int)
    return df


def add_cat_helpers(df: pd.DataFrame) -> pd.DataFrame:
    """Create string categoricals from numerics for PyTorch Forecasting."""
    df["Month_cat"]       = df["Month"].astype(str)
    df["day_of_week_cat"] = df["day_of_week"].astype(str)
    df["season_cat"]      = df["season_number"].astype(str)
    return df


def load_and_prepare(path: str, min_dt=None) -> tuple[pd.DataFrame, pd.Timestamp]:
    df = pd.read_parquet(path).reset_index(drop=True)
    df, _ = optimize_df_for_memory(df)

    # Cast everything back to float32 for PyTorch Forecasting compatibility
    for col in df.select_dtypes(include=["int8","int16","int32","uint8","uint16","uint32"]).columns:
        if col not in ["Year","Month","Day","Hour","day_of_week","season_number"]:
            df[col] = df[col].astype("float32")

    df = recover_station_id(df)
    df = recover_ohe_cat(df, prefix="Тип_",    new_col="Тип_cat")
    df = recover_ohe_cat(df, prefix="Область_", new_col="Область_cat")
    df = build_time_index(df)
    df = add_cat_helpers(df)

    # Ensure target is float32
    df[Y_COL] = df[Y_COL].astype("float32")

    # Sort for correct sequence ordering
    df = df.sort_values([GROUP_COL, "time_idx"]).reset_index(drop=True)

    return df


print("Loading train …")
train = load_and_prepare(TRAIN_PATH)

# Compute global min datetime from train so val/test share the same time_idx origin
GLOBAL_MIN_DT = train["datetime"].min()

print("Loading val   …")
val = load_and_prepare(VAL_PATH)

print("Loading test  …")
test = load_and_prepare(TEST_PATH)

# Align time_idx to the same global origin across splits
for split in [val, test]:
    split["time_idx"] = ((split["datetime"] - GLOBAL_MIN_DT).dt.total_seconds() / 3600).astype(int)

print(f"\nTrain shape : {train.shape}")
print(f"Val   shape : {val.shape}")
print(f"Test  shape : {test.shape}")
print(f"\nStations in train : {train[GROUP_COL].nunique()}")
print(f"time_idx range    : [{train['time_idx'].min()}, {train['time_idx'].max()}]")

#%%
# ==============================================================================
# Cell 6 – Build PyTorch Forecasting Datasets
# ==============================================================================

# Combine train + val so the val dataset can reference the encoder look-back
# that might overlap with train rows.  We use the standard pattern:
#   training_cutoff = max time_idx in train
training_cutoff = train["time_idx"].max()

# Concatenate for creating the datasets (PTF handles the split internally)
data_full = pd.concat([train, val], ignore_index=True).sort_values(
    [GROUP_COL, "time_idx"]
).reset_index(drop=True)

# ── Sanity-check required columns exist ────────────────────────────────────────
missing = [
    c for c in FUTURE_REALS + PAST_REALS + STATIC_REALS
    + TIME_VARYING_KNOWN_CATS + STATIC_CATS
    if c not in data_full.columns
]
if missing:
    print("⚠️  Missing columns (remove from config or add to pre-processing):")
    for m in missing:
        print("   ", m)
else:
    print("✅  All configured columns present.")

# Filter to only existing columns
def filter_existing(lst, df):
    return [c for c in lst if c in df.columns]

future_reals_ok       = filter_existing(FUTURE_REALS, data_full)
past_reals_ok         = filter_existing(PAST_REALS,   data_full)
static_reals_ok       = filter_existing(STATIC_REALS, data_full)
tv_known_cats_ok      = filter_existing(TIME_VARYING_KNOWN_CATS, data_full)
static_cats_ok        = filter_existing(STATIC_CATS,  data_full)

# ── Training dataset ───────────────────────────────────────────────────────────
training = TimeSeriesDataSet(
    data_full[data_full["time_idx"] <= training_cutoff],
    time_idx                  = "time_idx",
    target                    = Y_COL,
    group_ids                 = [GROUP_COL],
    min_encoder_length        = MAX_ENCODER_LENGTH // 2,   # allow shorter look-backs
    max_encoder_length        = MAX_ENCODER_LENGTH,
    min_prediction_length     = 1,
    max_prediction_length     = MAX_PREDICTION_LENGTH,
    static_categoricals       = static_cats_ok,
    static_reals              = static_reals_ok,
    time_varying_known_reals  = future_reals_ok + ["time_idx"],
    time_varying_known_categoricals = tv_known_cats_ok,
    time_varying_unknown_reals = [Y_COL] + past_reals_ok,
    target_normalizer         = GroupNormalizer(
        groups=[GROUP_COL], transformation="softplus"
    ),
    add_relative_time_idx     = True,
    add_target_scales         = True,
    add_encoder_length        = True,
    allow_missing_timesteps   = True,   # stations may have gaps
)

# ── Validation dataset – shares the same parameters as training ────────────────
validation = TimeSeriesDataSet.from_dataset(
    training,
    data_full,
    predict=True,           # only the prediction window beyond training_cutoff
    stop_randomization=True,
)

# ── DataLoaders ────────────────────────────────────────────────────────────────
NUM_WORKERS = 4  # adjust to your CPU count

train_loader = training.to_dataloader(
    train=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
)
val_loader = validation.to_dataloader(
    train=False, batch_size=BATCH_SIZE * 2, num_workers=NUM_WORKERS, pin_memory=True
)

print(f"\nTraining   batches : {len(train_loader)}")
print(f"Validation batches : {len(val_loader)}")

#%%
# ==============================================================================
# Cell 7 – Build the TFT / LSTM Model
# ==============================================================================
# PyTorch Forecasting's TemporalFusionTransformer uses LSTMs internally as
# its sequence encoder/decoder – it is the recommended "LSTM +" architecture
# for multi-series panel forecasting with covariates.
# If you strictly need a vanilla LSTM, swap TFT for
# pytorch_forecasting.models.DeepAR or pytorch_forecasting.models.NHiTS.

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate          = LEARNING_RATE,
    hidden_size            = HIDDEN_SIZE,
    attention_head_size    = ATTENTION_HEAD_SIZE,
    dropout                = DROPOUT,
    hidden_continuous_size = HIDDEN_CONTINUOUS_SIZE,
    loss                   = QuantileLoss(),          # change to SMAPE() for point forecast
    log_interval           = 10,
    optimizer              = "adam",
    reduce_on_plateau_patience = 4,
)

print(f"Number of parameters: {tft.size() / 1e3:.1f}k")

#%%
# ==============================================================================
# Cell 8 – Callbacks & Trainer
# ==============================================================================

early_stop = EarlyStopping(
    monitor   = "val_loss",
    min_delta = 1e-4,
    patience  = 10,
    verbose   = True,
    mode      = "min",
)

lr_logger = LearningRateMonitor()

checkpoint = ModelCheckpoint(
    dirpath   = "checkpoints/",
    filename  = "tft-{epoch:02d}-{val_loss:.4f}",
    monitor   = "val_loss",
    mode      = "min",
    save_top_k = 2,
)

logger = TensorBoardLogger("tb_logs", name="tft_gas_stations")

trainer = pl.Trainer(
    max_epochs           = MAX_EPOCHS,
    accelerator          = "gpu" if torch.cuda.is_available() else "cpu",
    devices              = 1,
    gradient_clip_val    = 0.1,
    callbacks            = [early_stop, lr_logger, checkpoint],
    logger               = logger,
    enable_progress_bar  = True,
    log_every_n_steps    = 10,
)

#%%
# ==============================================================================
# Cell 9 – Train
# ==============================================================================

trainer.fit(
    tft,
    train_dataloaders = train_loader,
    val_dataloaders   = val_loader,
)

print(f"\nBest checkpoint : {checkpoint.best_model_path}")
print(f"Best val_loss   : {checkpoint.best_model_score:.6f}")

#%%
# ==============================================================================
# Cell 10 – Load Best Checkpoint & Validate
# ==============================================================================

best_tft = TemporalFusionTransformer.load_from_checkpoint(checkpoint.best_model_path)
best_tft.eval()

# Predict on validation set – returns a dict with 'prediction' tensor (quantiles)
val_raw_predictions, val_index = best_tft.predict(
    val_loader,
    mode            = "raw",
    return_index    = True,
    trainer_kwargs  = {"accelerator": "gpu" if torch.cuda.is_available() else "cpu"},
)

# Use median quantile (index 3 out of 7 for default QuantileLoss)
# If you switched to a point-loss (SMAPE), val_raw_predictions["prediction"] is already 1-D per step.
val_pred_quantiles = val_raw_predictions["prediction"]
val_pred = val_pred_quantiles[..., 3].cpu().numpy().flatten()   # median

# Align ground truth from the index
val_true = (
    data_full
    .merge(val_index, on=[GROUP_COL, "time_idx"], how="inner")[Y_COL]
    .values
)

print("\n── Validation Metrics ──────────────────────────────────────")
print(f"SMAPE : {smape(val_true, val_pred):.4f}")
print(f"RMSE  : {rmse(val_true,  val_pred):.4f}")
print(f"MAPE  : {mape(val_true,  val_pred):.2f} %")

#%%
# ==============================================================================
# Cell 11 – Build Test Dataset & Run Inference
# ==============================================================================

# The test dataset is built from the FULL data (train + val + test) but we only
# predict beyond the validation cutoff.
val_cutoff = data_full["time_idx"].max()

data_all = pd.concat([data_full, test], ignore_index=True).sort_values(
    [GROUP_COL, "time_idx"]
).reset_index(drop=True)

test_dataset = TimeSeriesDataSet.from_dataset(
    training,
    data_all,
    predict=True,
    stop_randomization=True,
)

test_loader = test_dataset.to_dataloader(
    train=False, batch_size=BATCH_SIZE * 2, num_workers=NUM_WORKERS, pin_memory=True
)

test_raw_predictions, test_index = best_tft.predict(
    test_loader,
    mode            = "raw",
    return_index    = True,
    trainer_kwargs  = {"accelerator": "gpu" if torch.cuda.is_available() else "cpu"},
)

test_pred_quantiles = test_raw_predictions["prediction"]
test_pred = test_pred_quantiles[..., 3].cpu().numpy().flatten()

test_true = (
    data_all
    .merge(test_index, on=[GROUP_COL, "time_idx"], how="inner")[Y_COL]
    .values
)

print("\n── Test Metrics ────────────────────────────────────────────")
print(f"SMAPE : {smape(test_true, test_pred):.4f}")
print(f"RMSE  : {rmse(test_true,  test_pred):.4f}")
print(f"MAPE  : {mape(test_true,  test_pred):.2f} %")

#%%
# ==============================================================================
# Cell 12 – Interpretation: Variable Importance
# ==============================================================================
import matplotlib.pyplot as plt

# Use a subset of validation data for speed
interpretation = best_tft.interpret_output(
    val_raw_predictions, reduction="sum"
)

best_tft.plot_interpretation(interpretation)
plt.tight_layout()
plt.savefig("variable_importance.png", dpi=150)
plt.show()
print("Variable importance plot saved to variable_importance.png")

#%%
# ==============================================================================
# Cell 13 – Per-station Forecast Plot (sample 4 stations)
# ==============================================================================

sample_stations = data_full[GROUP_COL].unique()[:4]

fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=False)

for ax, station in zip(axes, sample_stations):
    # Filter index to this station
    mask = val_index[GROUP_COL] == station
    if mask.sum() == 0:
        ax.set_title(f"{station} – no val data")
        continue

    idx_rows = val_index[mask]
    true_vals = (
        data_full.merge(idx_rows, on=[GROUP_COL, "time_idx"], how="inner")[Y_COL].values
    )
    pred_vals = val_pred_quantiles[mask.values, :, 3].cpu().numpy().flatten()

    t = np.arange(len(true_vals))
    ax.plot(t, true_vals, label="Actual",    color="#1f77b4")
    ax.plot(t, pred_vals, label="Predicted", color="#ff7f0e", linestyle="--")
    ax.set_title(station)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylabel(Y_COL)

plt.suptitle("Validation Forecasts – Sample Stations", y=1.01, fontsize=13)
plt.tight_layout()
plt.savefig("sample_forecasts.png", dpi=150)
plt.show()
print("Sample forecast plot saved to sample_forecasts.png")

#%%
# ==============================================================================
# Cell 14 – Save Predictions to CSV
# ==============================================================================

val_results = val_index.copy()
val_results["pred_median"] = val_pred
val_results["actual"]      = val_true
val_results.to_csv("val_predictions.csv", index=False)

test_results = test_index.copy()
test_results["pred_median"] = test_pred
test_results["actual"]      = test_true
test_results.to_csv("test_predictions.csv", index=False)

print("val_predictions.csv  written.")
print("test_predictions.csv written.")
#%%