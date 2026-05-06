"""
Inference script for the fine-tuned Chronos-2 model.

Mirrors the structure of predict_xgboost(...) — same signature, same return
columns — but runs rolling multivariate forecasting with Chronos-2 instead of
a flat regressor.

Loads weights from:
    models/chronos-2-finetuned/config.json
    models/chronos-2-finetuned/model.safetensors
"""

import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

try:
    # Preferred path (chronos-forecasting >= 2.1.0)
    from chronos import Chronos2Pipeline
except ImportError:
    try:
        # Fallback for some pre-release / dev installs
        from chronos.chronos2.pipeline import Chronos2Pipeline
    except ImportError as e:
        raise ImportError(
            "Could not import Chronos2Pipeline. The installed `chronos-forecasting` "
            "is too old — Chronos-2 was added in 2.1.0. Upgrade with:\n\n"
            "    pip install --upgrade 'chronos-forecasting>=2.1.0'\n"
        ) from e

from inference_data_preprocessing import inference_preprocessing_full

# -----------------------------------------------------------------------------
# Constants — must match the training notebook exactly so the fine-tuned model
# sees inputs in the layout it was trained on.
# -----------------------------------------------------------------------------
Y_COL         = "sum_of_kWh"
GROUP_COL     = "eic_code"
TIME_IDX_COL  = "time_idx"

MODEL_DIR     = "models/chronos-2-finetuned"

CONTEXT_LEN          = 512
PREDICTION_LEN       = 64
QUANTILE_LEVELS      = [0.1, 0.5, 0.9]
INFERENCE_BATCH_SIZE = 16

# Feature lists — copied verbatim from the training notebook.
weather_cols_to_keep = [
    "temperature_2m",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "shortwave_radiation",
    "direct_normal_irradiance",
]
other_cols = [
    "dam_price", "buy_bm_price", "sell_bm_price",
    "max_power", "max_solar", "max_ev",
]
calendar_cols       = ["Month", "Day", "Hour", "day_of_week", "season"]
STATIC_NUMERIC_COLS = ["latitude", "longitude"]
STATIC_CAT_COLS     = ["dso_desc", "station_type", "oblast"]

NUMERIC_COVARIATES     = weather_cols_to_keep + other_cols + calendar_cols + STATIC_NUMERIC_COLS
CATEGORICAL_COVARIATES = STATIC_CAT_COLS
ALL_COVARIATES         = NUMERIC_COVARIATES + CATEGORICAL_COVARIATES

# Chronos predict_df naming
CHRONOS_ID_COL  = "item_id"
CHRONOS_TS_COL  = "timestamp"
CHRONOS_TGT_COL = "target"

# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------------------------------
# Helpers (lifted from the training notebook so inference matches training)
# -----------------------------------------------------------------------------
def _strip_tz(s: pd.Series) -> pd.Series:
    """Drop tz info if present (deprecation-safe)."""
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        return s.dt.tz_localize(None)
    return s


def _regularize_series(g: pd.DataFrame, freq: str = "h") -> pd.DataFrame:
    """Reindex one station's frame onto a complete `freq` grid.

    Chronos-2's `predict_df` calls `pd.infer_freq` on every series and rejects
    anything irregular: duplicate timestamps (DST autumn fold), missing hours
    (DST spring forward, gaps in the source data), or out-of-order rows all
    cause `Could not infer frequency for series ...`.

    We therefore:
      1) drop exact duplicate (id, datetime) rows (keep last),
      2) reindex onto pd.date_range(min, max, freq=freq),
      3) forward-then-back fill covariates and static columns,
      4) leave the target NaN where it was originally missing — Chronos-2
         tolerates NaN targets in context.
    """
    g = g.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
    full_idx = pd.date_range(g["datetime"].min(), g["datetime"].max(), freq=freq)
    g = g.set_index("datetime").reindex(full_idx)
    g.index.name = "datetime"

    # The id column must be filled (it's constant per group).
    # Guard against pandas 2.3+ which excludes the grouping column from the
    # group DataFrame inside apply — handled by the caller restoring it.
    if GROUP_COL in g.columns:
        g[GROUP_COL] = g[GROUP_COL].ffill().bfill()

    # Static / categorical covariates: ffill/bfill (constant per station)
    for c in CATEGORICAL_COVARIATES + STATIC_NUMERIC_COLS:
        if c in g.columns:
            g[c] = g[c].ffill().bfill()

    # Time-varying numeric covariates: ffill then bfill to plug small gaps
    time_varying = [c for c in NUMERIC_COVARIATES if c not in STATIC_NUMERIC_COLS]
    for c in time_varying:
        if c in g.columns:
            g[c] = g[c].ffill().bfill()

    # time_idx is just a row counter — recreate it to stay monotonically increasing
    if TIME_IDX_COL in g.columns:
        g[TIME_IDX_COL] = np.arange(len(g))

    return g.reset_index()


def _to_chronos_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to the dtypes Chronos-2 expects and put each series on a
    regular hourly grid so `pd.infer_freq` always succeeds."""
    out = df[[GROUP_COL, "datetime", Y_COL, TIME_IDX_COL] + ALL_COVARIATES].copy()
    out["datetime"] = _strip_tz(out["datetime"])

    # Regularize each station onto a full hourly grid before dtype casting.
    # Explicit loop instead of groupby.apply: pandas 2.3+ excludes the grouping
    # column from the group DataFrame inside apply, which breaks _regularize_series.
    parts = []
    for code, group in out.groupby(GROUP_COL, sort=False):
        reg = _regularize_series(group.copy())
        if GROUP_COL not in reg.columns:
            reg[GROUP_COL] = code  # restore if pandas stripped it
        parts.append(reg)
    out = pd.concat(parts, ignore_index=True)

    for c in NUMERIC_COVARIATES:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(np.float32)

    for c in CATEGORICAL_COVARIATES:
        out[c] = out[c].astype(str)

    # Keep target as float32; NaNs are preserved for the future portion.
    out[Y_COL] = pd.to_numeric(out[Y_COL], errors="coerce").astype(np.float32)
    return out


def _rename_for_chronos(df: pd.DataFrame, drop_target: bool = False) -> pd.DataFrame:
    cols = {GROUP_COL: CHRONOS_ID_COL, "datetime": CHRONOS_TS_COL, Y_COL: CHRONOS_TGT_COL}
    out = df.rename(columns=cols)
    if drop_target and CHRONOS_TGT_COL in out.columns:
        out = out.drop(columns=[CHRONOS_TGT_COL])
    keep = [CHRONOS_ID_COL, CHRONOS_TS_COL] + ALL_COVARIATES
    if not drop_target:
        keep.insert(2, CHRONOS_TGT_COL)
    return out[keep]


def _resolve_median_col(pred_df: pd.DataFrame):
    for cand in pred_df.columns:
        if str(cand) == "0.5":
            return cand
    raise KeyError(
        f"Could not find median (0.5) column in predict_df output. "
        f"Got columns: {list(pred_df.columns)}"
    )


# -----------------------------------------------------------------------------
# Pipeline loader (cached so repeat calls don't reload weights)
# -----------------------------------------------------------------------------
_PIPELINE_CACHE: Dict[str, Chronos2Pipeline] = {}


def _load_pipeline(model_dir: str = MODEL_DIR) -> Chronos2Pipeline:
    if model_dir in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[model_dir]

    model_path = Path(model_dir)
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(f"Missing {model_path / 'config.json'}")
    if not (model_path / "model.safetensors").exists():
        raise FileNotFoundError(f"Missing {model_path / 'model.safetensors'}")

    pipeline = Chronos2Pipeline.from_pretrained(
        str(model_path),
        device_map=DEVICE,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    _PIPELINE_CACHE[model_dir] = pipeline
    return pipeline


# -----------------------------------------------------------------------------
# Rolling inference over the unknown-target tail
# -----------------------------------------------------------------------------
def _rolling_predict(
    pipeline: Chronos2Pipeline,
    df_chronos: pd.DataFrame,
    horizon_start: pd.Timestamp,
    horizon_end: pd.Timestamp,
    chunk: int = PREDICTION_LEN,
    context_len: int = CONTEXT_LEN,
) -> pd.DataFrame:
    """Roll Chronos-2 across the [horizon_start, horizon_end] window.

    Splits each station's frame into:
      - history:  rows with datetime < horizon_start (target may be observed
                  or NaN — Chronos tolerates NaN targets in context)
      - horizon:  rows with horizon_start <= datetime <= horizon_end
                  (rows we forecast; target is NaN here)

    We iterate `chunk`-sized windows over the horizon, calling predict_df with
    the most recent `context_len` history rows. Because we have no ground
    truth for the future, we extend the rolling history with the model's own
    median forecast.
    """
    history_by_code: Dict[str, pd.DataFrame] = {}
    horizon_by_code: Dict[str, pd.DataFrame] = {}
    for code, g in df_chronos.groupby(GROUP_COL, sort=False):
        g = g.sort_values("datetime").reset_index(drop=True)
        is_horizon = (g["datetime"] >= horizon_start) & (g["datetime"] <= horizon_end)
        hist_part = g[~is_horizon].reset_index(drop=True)
        hor_part  = g[is_horizon].reset_index(drop=True)

        if len(hist_part) == 0 or len(hor_part) == 0:
            # Need at least some context and some horizon to forecast.
            continue
        # Force the target to NaN on the horizon side (defensive — should
        # already be NaN, but if upstream filled anything we don't want
        # leakage).
        hor_part = hor_part.copy()
        hor_part[Y_COL] = np.float32("nan")

        history_by_code[code] = hist_part
        horizon_by_code[code] = hor_part

    codes = list(horizon_by_code.keys())
    if not codes:
        return pd.DataFrame(columns=[GROUP_COL, "datetime", "pred"])

    # Mutable working history we extend with median predictions as we roll.
    work_hist = {c: history_by_code[c].copy() for c in codes}
    horizon_lens = {c: len(horizon_by_code[c]) for c in codes}
    H = max(horizon_lens.values())
    n_chunks = math.ceil(H / chunk)

    out_rows: List[pd.DataFrame] = []
    cursor = 0
    pbar = tqdm(total=n_chunks, desc="chronos rolling")
    while cursor < H:
        # Bucket stations by remaining-tail length — predict_df needs uniform
        # prediction_length per call.
        by_steps: Dict[int, List[str]] = {}
        for c in codes:
            remaining = horizon_lens[c] - cursor
            if remaining <= 0:
                continue
            steps = min(chunk, remaining)
            by_steps.setdefault(steps, []).append(c)

        if not by_steps:
            break

        for steps, group_codes in by_steps.items():
            ctx_frames, fut_frames = [], []
            for c in group_codes:
                full_hist = work_hist[c]
                future_slice = horizon_by_code[c].iloc[cursor : cursor + steps]
                ctx = full_hist.tail(context_len)
                ctx_frames.append(ctx)
                fut_frames.append(future_slice)

            ctx_df = pd.concat(ctx_frames, ignore_index=True)
            fut_df = pd.concat(fut_frames, ignore_index=True)

            ctx_long = _rename_for_chronos(ctx_df, drop_target=False)
            fut_long = _rename_for_chronos(fut_df, drop_target=True)

            forecast_chunks = []
            for i in range(0, len(group_codes), INFERENCE_BATCH_SIZE):
                batch_ids = group_codes[i : i + INFERENCE_BATCH_SIZE]
                ctx_b = ctx_long[ctx_long[CHRONOS_ID_COL].isin(batch_ids)]
                fut_b = fut_long[fut_long[CHRONOS_ID_COL].isin(batch_ids)]
                pred_b = pipeline.predict_df(
                    ctx_b,
                    future_df=fut_b,
                    prediction_length=steps,
                    quantile_levels=QUANTILE_LEVELS,
                    id_column=CHRONOS_ID_COL,
                    timestamp_column=CHRONOS_TS_COL,
                    target=CHRONOS_TGT_COL,
                )
                forecast_chunks.append(pred_b)
            pred_df = pd.concat(forecast_chunks, ignore_index=True)

            median_col = _resolve_median_col(pred_df)
            pred_df = pred_df.rename(columns={
                CHRONOS_ID_COL: GROUP_COL,
                CHRONOS_TS_COL: "datetime",
                median_col: "pred",
            })
            pred_df["pred"] = pred_df["pred"].clip(lower=0)

            out_rows.append(pred_df[[GROUP_COL, "datetime", "pred"]])

            # Extend history with the model's own predictions so the next chunk
            # has continuous coverage. We replace Y_COL on the future slice
            # with the median forecast and append.
            for c in group_codes:
                new_obs = horizon_by_code[c].iloc[cursor : cursor + steps].copy()
                preds_for_c = (
                    pred_df[pred_df[GROUP_COL] == c]
                    .set_index("datetime")["pred"]
                )
                new_obs[Y_COL] = new_obs["datetime"].map(preds_for_c).astype(np.float32)
                work_hist[c] = pd.concat([work_hist[c], new_obs], ignore_index=True)

        cursor += chunk
        pbar.update(1)

    pbar.close()
    return pd.concat(out_rows, ignore_index=True)


# -----------------------------------------------------------------------------
# Public API — analogous to predict_xgboost
# -----------------------------------------------------------------------------
def predict_chronos_ft(df: pd.DataFrame, start_dt: str, end_dt: str) -> pd.DataFrame:
    """Run rolling Chronos-2 inference on the unknown-target portion of df.

    Returns a DataFrame with the same columns predict_xgboost returns:
        [eic_code, datetime, sum_of_kWh, dam_price, buy_bm_price, sell_bm_price]
    where sum_of_kWh holds the median forecast.
    """
    print("Loading data...")
    df = inference_preprocessing_full(df, start_dt, end_dt)

    print("Loading Chronos-2 pipeline...")
    pipeline = _load_pipeline(MODEL_DIR)

    df_chronos = (
        _to_chronos_frame(df)
        .sort_values([GROUP_COL, "datetime"])
        .reset_index(drop=True)
    )

    horizon_start = pd.Timestamp(start_dt)
    horizon_end   = pd.Timestamp(end_dt)
    # If end_dt is a bare date (00:00:00), treat the horizon as inclusive of
    # the whole final day to match how XGBoost's mask (Y_COL.isna()) covers it.
    if horizon_end.normalize() == horizon_end:
        horizon_end = horizon_end + pd.Timedelta(hours=23, minutes=59)

    print("Running rolling inference...")
    preds = _rolling_predict(pipeline, df_chronos, horizon_start, horizon_end)

    # Same shape predict_xgboost returns: rows from the original df where Y was
    # NaN, with predictions merged in by (code, datetime). Using the original
    # df preserves any irregular timestamps the source had.
    predict_df = df[df[Y_COL].isna()].copy()
    predict_df["datetime"] = _strip_tz(predict_df["datetime"])
    predict_df = predict_df.merge(
        preds[[GROUP_COL, "datetime", "pred"]],
        on=[GROUP_COL, "datetime"],
        how="left",
    )
    predict_df[Y_COL] = predict_df["pred"].clip(lower=0)

    return predict_df[[GROUP_COL, "datetime", Y_COL,
                       "dam_price", "buy_bm_price", "sell_bm_price"
                       ]]


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_excel("../data/inference/one_station_sample.xlsx")
    df["Дата"] = pd.to_datetime(df["Дата"])
    df_preds = predict_chronos_ft(df, start_dt="2025-09-01", end_dt="2025-10-30")
    print(df_preds)