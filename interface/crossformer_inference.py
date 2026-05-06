"""
Crossformer inference script — analogous to predict_xgboost.

Loads a trained Crossformer checkpoint (models/crossformer_best.pt) and
produces 48-hour forecasts for the rows in the dataframe whose target is
NaN, using a sliding window of SEQ_LEN historical hours per station.

Public entry point:
    predict_crossformer(df, start_dt, end_dt) -> pd.DataFrame
"""

import os
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from inference_data_preprocessing import inference_preprocessing_full

# ──────────────────────────────────────────────────────────────────────────────
# Constants — must match training (train_crossformer_v3.ipynb)
# ──────────────────────────────────────────────────────────────────────────────
Y_COL        = "sum_of_kWh"
GROUP_COL    = "eic_code"
TIME_IDX_COL = "time_idx"

weather_cols_to_drop = [
       'apparent_temperature', 'dew_point_2m', 'relative_humidity_2m',
    'rain', 'snowfall', 'cloud_cover_low',
       'cloud_cover_mid', 'cloud_cover_high', 'surface_pressure',
    'wind_direction_10m', 'wind_gusts_10m', 'diffuse_radiation',
]

weather_cols_to_keep = [
    "temperature_2m",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "shortwave_radiation",
    "direct_normal_irradiance"
]

WEATHER_FEATS     = weather_cols_to_keep
STATIC_FEATS_REAL = ["latitude", "longitude"]
TIME_FEATS        = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
PRICE_COLS        = ["dam_price", "sell_bm_price", "buy_bm_price"]

# Target FIRST so the model's RevIN denormalization (index 0) is correct.
# NOTE: prices are NOT model inputs. They are only used by the training loss
# (and carried through to the output frame at inference). The checkpoint was
# trained with D_IN = 1 + 6 + 2 + 6 = 15.
INPUT_FEATURES = [Y_COL] + WEATHER_FEATS + STATIC_FEATS_REAL + TIME_FEATS
D_IN = len(INPUT_FEATURES)            # 15

# Sequence
SEQ_LEN  = 168
PRED_LEN = 48
SEG_LEN  = 12

# Model hyperparameters
D_MODEL  = 256
N_HEADS  = 4
E_LAYERS = 3
D_FF     = 512
DROPOUT  = 0.1
WIN_SIZE = 2

CKPT_PATH = "models/crossformer_best.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Cyclical time features (verbatim from training)
# ──────────────────────────────────────────────────────────────────────────────
def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["datetime"])
    df["hour_sin"]  = np.sin(2 * np.pi * dt.dt.hour      / 24).astype("float32")
    df["hour_cos"]  = np.cos(2 * np.pi * dt.dt.hour      / 24).astype("float32")
    df["dow_sin"]   = np.sin(2 * np.pi * dt.dt.dayofweek / 7 ).astype("float32")
    df["dow_cos"]   = np.cos(2 * np.pi * dt.dt.dayofweek / 7 ).astype("float32")
    df["month_sin"] = np.sin(2 * np.pi * dt.dt.month     / 12).astype("float32")
    df["month_cos"] = np.cos(2 * np.pi * dt.dt.month     / 12).astype("float32")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CROSSFORMER MODEL — verbatim from training notebook
# ══════════════════════════════════════════════════════════════════════════════
class DSW_Embedding(nn.Module):
    def __init__(self, seg_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.seg_len = seg_len
        self.proj = nn.Linear(seg_len, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        pad = (self.seg_len - T % self.seg_len) % self.seg_len
        if pad:
            x = F.pad(x, (0, 0, pad, 0))
        num_seg = x.shape[1] // self.seg_len
        x = x.reshape(B, num_seg, self.seg_len, D).permute(0, 3, 1, 2)
        return self.drop(self.proj(x))


class TwoStageAttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ct_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ct_norm1 = nn.LayerNorm(d_model)
        self.ct_ff    = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.ct_norm2 = nn.LayerNorm(d_model)
        self.cd_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cd_norm1 = nn.LayerNorm(d_model)
        self.cd_ff    = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.cd_norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, D, L, d = x.shape
        xt = x.reshape(B * D, L, d)
        out, _ = self.ct_attn(xt, xt, xt)
        xt = self.ct_norm1(xt + self.drop(out))
        xt = self.ct_norm2(xt + self.ct_ff(xt))
        x  = xt.reshape(B, D, L, d)
        xd = x.permute(0, 2, 1, 3).reshape(B * L, D, d)
        out, _ = self.cd_attn(xd, xd, xd)
        xd = self.cd_norm1(xd + self.drop(out))
        xd = self.cd_norm2(xd + self.cd_ff(xd))
        x  = xd.reshape(B, L, D, d).permute(0, 2, 1, 3)
        return x


class SegMerging(nn.Module):
    def __init__(self, d_model: int, win_size: int = 2):
        super().__init__()
        self.win = win_size
        self.norm = nn.LayerNorm(d_model * win_size)
        self.proj = nn.Linear(d_model * win_size, d_model)

    def forward(self, x):
        B, D, L, d = x.shape
        if L % self.win:
            x = F.pad(x, (0, 0, 0, self.win - L % self.win))
            L = x.shape[2]
        x = x.reshape(B, D, L // self.win, self.win * d)
        return self.proj(self.norm(x))


class CrossformerEncoder(nn.Module):
    def __init__(self, attn_layers, merge_layers):
        super().__init__()
        self.attn = attn_layers
        self.merge = merge_layers

    def forward(self, x):
        outs = []
        for i, layer in enumerate(self.attn):
            x = layer(x)
            outs.append(x)
            if i < len(self.merge):
                x = self.merge[i](x)
        return outs


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.self_norm  = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.ff_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mem):
        B, D, Lq, d = x.shape
        _, _, Lk, _ = mem.shape
        xt = x.reshape(B * D, Lq, d)
        sa, _ = self.self_attn(xt, xt, xt)
        xt = self.self_norm(xt + self.drop(sa))
        mk = mem.reshape(B * D, Lk, d)
        ca, _ = self.cross_attn(xt, mk, mk)
        xt = self.cross_norm(xt + self.drop(ca))
        xt = self.ff_norm(xt + self.ff(xt))
        return xt.reshape(B, D, Lq, d)


class Crossformer(nn.Module):
    def __init__(self, seq_len, pred_len, seg_len, d_in,
                 d_model=256, n_heads=4, e_layers=3, d_ff=512,
                 dropout=0.1, win_size=2):
        super().__init__()
        self.pred_len = pred_len
        self.seg_len  = seg_len
        self.embed = DSW_Embedding(seg_len, d_model, dropout)

        pad_in  = (seg_len - seq_len  % seg_len) % seg_len
        pad_out = (seg_len - pred_len % seg_len) % seg_len
        self.in_seg_num  = (seq_len  + pad_in)  // seg_len
        self.out_seg_num = (pred_len + pad_out) // seg_len

        self.enc_pos = nn.Parameter(torch.randn(1, 1, self.in_seg_num,  d_model) * 0.02)
        self.dec_pos = nn.Parameter(torch.randn(1, 1, self.out_seg_num, d_model) * 0.02)
        self.dim_pos = nn.Parameter(torch.randn(1, d_in, 1,             d_model) * 0.02)

        self.encoder = CrossformerEncoder(
            nn.ModuleList([TwoStageAttentionLayer(d_model, n_heads, d_ff, dropout)
                           for _ in range(e_layers)]),
            nn.ModuleList([SegMerging(d_model, win_size) for _ in range(e_layers - 1)]),
        )
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)
        ])
        self.out_proj = nn.Linear(d_model, seg_len)

    def forward(self, x):
        B, T, D = x.shape
        mu  = x.mean(dim=1, keepdim=True)
        sig = x.std (dim=1, keepdim=True).clamp(1e-5)
        x_n = (x - mu) / sig

        enc_in   = self.embed(x_n) + self.enc_pos + self.dim_pos
        enc_outs = self.encoder(enc_in)

        dec_q = (self.dec_pos + self.dim_pos).expand(B, -1, -1, -1)
        for dec_layer, mem in zip(self.dec_layers, reversed(enc_outs)):
            dec_q = dec_layer(dec_q, mem)

        out = self.out_proj(dec_q)
        out = out.reshape(B, D, -1)
        out = out[:, 0, :self.pred_len]
        out = out * sig[:, 0, 0:1] + mu[:, 0, 0:1]
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Model loader (cached so repeated calls don't reload weights)
# ──────────────────────────────────────────────────────────────────────────────
_MODEL_CACHE = {"model": None, "ckpt": None}


def _load_model(ckpt_path: str = CKPT_PATH) -> Crossformer:
    if _MODEL_CACHE["model"] is not None and _MODEL_CACHE["ckpt"] == ckpt_path:
        return _MODEL_CACHE["model"]

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Crossformer checkpoint not found: {ckpt_path}")

    model = Crossformer(
        seq_len=SEQ_LEN, pred_len=PRED_LEN, seg_len=SEG_LEN, d_in=D_IN,
        d_model=D_MODEL, n_heads=N_HEADS, e_layers=E_LAYERS,
        d_ff=D_FF, dropout=DROPOUT, win_size=WIN_SIZE,
    ).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["ckpt"]  = ckpt_path
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────
def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time features and drop unused weather columns. Mirrors
    the in-notebook preprocessing applied to `all_data` before windowing."""
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df["datetime"] = df["datetime"].dt.tz_convert(None)

    df = add_cyclical_time_features(df)

    drop_cols = [c for c in weather_cols_to_drop if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Sanity-check feature presence
    missing = [f for f in INPUT_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required input features: {missing}")
    return df


@torch.no_grad()
def _predict_per_station(model: Crossformer, df: pd.DataFrame) -> pd.DataFrame:
    """For every station, slide a SEQ_LEN window forward by PRED_LEN and
    forecast PRED_LEN steps.

    The forecast horizon is much longer than PRED_LEN, so we run autoregressively:
    every prediction is written back into the working feature array's target
    column, so the next window's lookback sees the model's own forecasts in
    place of the NaN target rows. Without this, the second-and-later windows
    see all-NaN target history, RevIN normalizes by ~0 std/mean, and outputs
    collapse to zero.

    Returns one row per originally-NaN target slot, with the same columns
    as predict_xgboost.
    """
    rows = []
    feat_cols  = INPUT_FEATURES
    price_cols = PRICE_COLS
    target_idx = feat_cols.index(Y_COL)   # always 0, but be explicit

    use_cuda = torch.cuda.is_available()

    for grp in tqdm(df[GROUP_COL].unique(), desc="Crossformer inference"):
        gdf = (df[df[GROUP_COL] == grp]
               .sort_values(TIME_IDX_COL)
               .reset_index(drop=True))

        # Drop duplicate time_idx rows (DST artifacts, duplicate inputs, etc.).
        # Without this, positional offsets diverge from time_idx offsets and
        # the window-contiguity check below rejects valid lookbacks.
        before_dedup = len(gdf)
        gdf = gdf.drop_duplicates(subset=[TIME_IDX_COL], keep="last").reset_index(drop=True)
        if len(gdf) != before_dedup:
            print(f"  [{grp}] dropped {before_dedup - len(gdf)} duplicate time_idx rows")

        # Working copy of features — column 0 (target) will be filled in
        # autoregressively as we predict.
        feat = gdf[feat_cols].values.astype(np.float32)
        prc  = gdf[price_cols].values.astype(np.float32)
        tidx = gdf[TIME_IDX_COL].values.astype(np.int64)
        dts  = gdf["datetime"].values
        y_isna_orig = gdf[Y_COL].isna().values.copy()   # mask of slots to PREDICT

        # Mask to remember which rows still need a prediction. Starts as the
        # original NaN mask, gets cleared as windows produce values.
        needs_pred = y_isna_orig.copy()

        # Sanity zero-fill of any non-target NaNs (weather/static/cyclical
        # should be fully populated by inference_preprocessing_full, but if
        # any sneak in, don't let them crash the forward pass).
        non_target_nan = np.isnan(feat[:, 1:]).any()
        if non_target_nan:
            feat[:, 1:] = np.nan_to_num(feat[:, 1:], nan=0.0)

        # Targets in the lookback that are NaN must be replaced with something
        # before any forward pass. Initialize them to the per-station mean of
        # the known history so the very first window's RevIN sees a sensible
        # scale even when its lookback overlaps the forecast region.
        known_targets = feat[~y_isna_orig, target_idx]
        if known_targets.size > 0:
            init_fill = float(np.nanmean(known_targets))
        else:
            init_fill = 0.0
        nan_target_rows = np.where(y_isna_orig)[0]
        feat[nan_target_rows, target_idx] = init_fill

        tmap = {int(t): i for i, t in enumerate(tidx)}

        nan_positions = np.where(y_isna_orig)[0]
        if len(nan_positions) == 0:
            continue

        first_nan_t = int(tidx[int(nan_positions[0])])
        last_nan_t  = int(tidx[int(nan_positions[-1])])

        t = first_nan_t
        while t <= last_nan_t:
            ctx_start = t - SEQ_LEN

            # Locate the lookback window. If ctx_start is outside the frame,
            # try to use the earliest SEQ_LEN-block we can: shift the forecast
            # anchor forward to whatever index does have SEQ_LEN of preceding
            # rows. (This only matters for the very first window when history
            # is shorter than SEQ_LEN.)
            if ctx_start not in tmap:
                if len(tidx) <= SEQ_LEN:
                    break  # not enough rows at all
                t = int(tidx[SEQ_LEN])
                ctx_start = t - SEQ_LEN
                if ctx_start not in tmap:
                    break

            if t not in tmap:
                # t fell into a gap; advance to the next available t.
                future_ts = tidx[tidx >= t]
                if len(future_ts) == 0:
                    break
                t = int(future_ts[0])
                ctx_start = t - SEQ_LEN
                if ctx_start not in tmap:
                    t += PRED_LEN
                    continue

            s = tmap[ctx_start]
            e = tmap[t]
            # Validate contiguity in the time-index sense: positions s..e-1 must
            # cover exactly SEQ_LEN distinct consecutive hours. Since we already
            # deduplicated by time_idx and sorted, this is equivalent to checking
            # that tidx jumps by 1 across the whole slice.
            if e - s != SEQ_LEN or int(tidx[e] - tidx[s]) != SEQ_LEN:
                # There's a real gap in the hourly timeline — skip this window.
                t += PRED_LEN
                continue

            window = feat[s:e]   # shape (SEQ_LEN, D_IN); already nan-clean

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
                xb = torch.from_numpy(window.copy()).unsqueeze(0).to(DEVICE)
                pred = model(xb).squeeze(0).float().cpu().numpy()
            pred = np.clip(pred, a_min=0.0, a_max=None)

            # Write predictions back into the working feature array (AR step)
            # AND emit output rows for the originally-NaN slots.
            for step in range(PRED_LEN):
                ti = t + step
                if ti not in tmap:
                    break
                pos = tmap[ti]

                # Always write predictions back into column 0 so subsequent
                # windows see them, but only OVERWRITE rows that were missing
                # in the original input. Real history stays intact.
                if y_isna_orig[pos]:
                    feat[pos, target_idx] = float(pred[step])

                if needs_pred[pos]:
                    needs_pred[pos] = False
                    rows.append({
                        GROUP_COL:        grp,
                        "datetime":       dts[pos],
                        Y_COL:            float(pred[step]),
                        "dam_price":      float(prc[pos, 0]),
                        "sell_bm_price":  float(prc[pos, 1]),
                        "buy_bm_price":   float(prc[pos, 2]),
                    })

            t += PRED_LEN

    return pd.DataFrame(rows, columns=[GROUP_COL, "datetime", Y_COL,
                                       "dam_price", "buy_bm_price", "sell_bm_price"])


def predict_crossformer(df: pd.DataFrame, start_dt, end_dt) -> pd.DataFrame:
    """Mirror of predict_xgboost. Returns predictions for the rows where the
    target is NaN, with the same columns as the XGBoost variant."""
    print("Loading data…")
    df = inference_preprocessing_full(df, start_dt, end_dt)
    df = _prepare_features(df)
    model = _load_model(CKPT_PATH)
    preds = _predict_per_station(model, df)
    # Match column order with predict_xgboost
    return preds[[GROUP_COL, "datetime", Y_COL,
                  "dam_price", "buy_bm_price", "sell_bm_price"]]


# ──────────────────────────────────────────────────────────────────────────────
# Test entry point — same shape as the XGBoost script
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_excel("../data/inference/one_station_sample.xlsx")
    df["Дата"] = pd.to_datetime(df["Дата"])
    df_preds = predict_crossformer(
        df,
        start_dt="2025-09-01",
        end_dt="2025-10-30",
    )
    print(df_preds)

    print(df_preds[Y_COL].describe())