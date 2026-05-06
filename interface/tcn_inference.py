"""
Inference script for the trained TCN (TemporalConvNet) model.

Mirrors the structure of the XGBoost inference script: takes a raw
DataFrame plus a [start_dt, end_dt] window, runs `inference_preprocessing_full`
on it, and returns predictions for rows where `sum_of_kWh` is missing.

The TCN relies on a `TimeSeriesDataSet` whose category encoders and target
normalizer must match the ones used at fit time. `pytorch_forecasting`
saves those parameters inside the Lightning checkpoint under
`hparams.dataset_parameters`, so we rebuild the dataset on inference data
via `TimeSeriesDataSet.from_parameters(...)` — no training file needed.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import copy
from typing import Optional, Union, List

from pytorch_forecasting import (
    TimeSeriesDataSet,
    BaseModelWithCovariates,
    MultiEmbedding,
)
from pytorch_forecasting.metrics import (
    MAE, SMAPE, RMSE, MAPE, MASE, MultiHorizonMetric,
)

from inference_data_preprocessing import inference_preprocessing_full


# ──────────────────────────────────────────────────────────────────────────
# Constants — must match the training notebook
# ──────────────────────────────────────────────────────────────────────────
Y_COL         = "sum_of_kWh"
GROUP_COL     = "eic_code"
TIME_IDX_COL  = "time_idx"
HORIZON       = 48
BATCH_SIZE    = 1024 * 2

CKPT_PATH   = "models/cnn-epoch=00-val_loss=8.0403.ckpt"
ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────────────────────────────────
# Model class — must be importable for `load_from_checkpoint` to work.
# Copied verbatim from the training notebook.
# ──────────────────────────────────────────────────────────────────────────
class _CausalConv1d(nn.Module):
    """Dilated causal 1-D convolution (no future leakage)."""
    def __init__(self, in_c: int, out_c: int, k: int, d: int):
        super().__init__()
        pad = (k - 1) * d
        self.conv  = nn.Conv1d(in_c, out_c, k, padding=pad, dilation=d)
        self.chomp = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return out[..., : -self.chomp] if self.chomp > 0 else out


class _TCNBlock(nn.Module):
    """Residual TCN block with two causal dilated convolutions."""
    def __init__(self, in_c: int, out_c: int, k: int, d: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            _CausalConv1d(in_c, out_c, k, d), nn.ReLU(), nn.Dropout(dropout),
            _CausalConv1d(out_c, out_c, k, d), nn.ReLU(), nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.net(x) + (x if self.skip is None else self.skip(x)))


class TemporalConvNet(BaseModelWithCovariates):
    def __init__(
        self,
        hidden_size:          int = 64,
        kernel_size:          int = 4,
        n_layers:             int = 4,
        dropout:              float = 0.1,
        max_prediction_length: int = 36,
        static_categoricals:               Optional[List[str]] = None,
        static_reals:                      Optional[List[str]] = None,
        time_varying_categoricals_encoder: Optional[List[str]] = None,
        time_varying_categoricals_decoder: Optional[List[str]] = None,
        categorical_groups:                Optional[dict]      = None,
        time_varying_reals_encoder:        Optional[List[str]] = None,
        time_varying_reals_decoder:        Optional[List[str]] = None,
        embedding_sizes:                   Optional[dict]      = None,
        embedding_paddings:                Optional[List[str]] = None,
        embedding_labels:                  Optional[dict]      = None,
        x_reals:                           Optional[List[str]] = None,
        x_categoricals:                    Optional[List[str]] = None,
        output_size:   Union[int, List[int]] = 1,
        target:        Union[str, List[str]] = None,
        loss:          MultiHorizonMetric    = None,
        logging_metrics: nn.ModuleList       = None,
        **kwargs,
    ):
        if loss            is None: loss            = RMSE()
        if logging_metrics is None: logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()])
        if static_categoricals               is None: static_categoricals               = []
        if static_reals                      is None: static_reals                      = []
        if time_varying_categoricals_encoder is None: time_varying_categoricals_encoder = []
        if time_varying_categoricals_decoder is None: time_varying_categoricals_decoder = []
        if categorical_groups                is None: categorical_groups                = {}
        if time_varying_reals_encoder        is None: time_varying_reals_encoder        = []
        if time_varying_reals_decoder        is None: time_varying_reals_decoder        = []
        if embedding_sizes                   is None: embedding_sizes                   = {}
        if embedding_paddings                is None: embedding_paddings                = []
        if embedding_labels                  is None: embedding_labels                  = {}
        if x_reals                           is None: x_reals                           = []
        if x_categoricals                    is None: x_categoricals                    = []

        self.save_hyperparameters()
        super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)

        self.embeddings = MultiEmbedding(
            embedding_sizes=embedding_sizes,
            embedding_paddings=embedding_paddings,
            categorical_groups=categorical_groups,
            x_categoricals=x_categoricals,
        )

        cont_size  = len(self.reals)
        cat_size   = sum(self.embeddings.output_size.values())
        input_size = cont_size + cat_size

        channels = [input_size] + [hidden_size] * n_layers
        self.tcn = nn.Sequential(*[
            _TCNBlock(channels[i], channels[i + 1], kernel_size, 2 ** i, dropout)
            for i in range(n_layers)
        ])

        out_sz = output_size if isinstance(output_size, int) else sum(output_size)
        self.output_projector = nn.Linear(hidden_size, max_prediction_length * out_sz)

    @classmethod
    def from_dataset(cls, dataset, allowed_encoder_known_variable_names=None, **kwargs):
        kwargs.setdefault("max_prediction_length", dataset.max_prediction_length)
        kwargs.setdefault("target", dataset.target)
        new_kwargs = copy(kwargs)
        new_kwargs.update(
            cls.deduce_default_output_parameters(dataset=dataset, kwargs=kwargs, default_loss=RMSE())
        )
        return super().from_dataset(
            dataset,
            allowed_encoder_known_variable_names=allowed_encoder_known_variable_names,
            **new_kwargs,
        )

    def forward(self, x: dict) -> dict:
        enc_cat  = x["encoder_cat"]
        enc_cont = x["encoder_cont"]

        if len(self.categoricals) > 0:
            embs     = self.embeddings(enc_cat)
            flat_emb = torch.cat(list(embs.values()), dim=-1)
            iv = flat_emb
        if len(self.reals) > 0:
            iv = enc_cont.clone()
        if len(self.reals) > 0 and len(self.categoricals) > 0:
            iv = torch.cat([enc_cont, flat_emb], dim=-1)

        tcn_out = self.tcn(iv.transpose(1, 2))
        last    = tcn_out[..., -1]

        raw    = self.output_projector(last)
        B      = raw.size(0)
        pred_l = self.hparams.max_prediction_length
        out_sz = (self.hparams.output_size
                  if isinstance(self.hparams.output_size, int)
                  else sum(self.hparams.output_size))
        pred   = raw.view(B, pred_l, out_sz)

        T_dec = x["decoder_cont"].size(1)
        pred  = pred[:, :T_dec, :]

        pred = self.transform_output(pred, target_scale=x["target_scale"])
        return self.to_network_output(prediction=pred)


# ──────────────────────────────────────────────────────────────────────────
# Lazy globals — load checkpoint once
# ──────────────────────────────────────────────────────────────────────────
_MODEL = None


def _ensure_loaded() -> TemporalConvNet:
    """Load the TCN checkpoint once and cache it."""
    global _MODEL
    if _MODEL is None:
        print(f"Loading TCN checkpoint: {CKPT_PATH}")
        # load_from_checkpoint calls model.to(device) after loading, which
        # triggers torchmetrics.Metric._apply to create a dummy CUDA tensor
        # (torch.zeros(1, device=self.device)) even when map_location="cpu".
        # This crashes on CPU-only torch builds. Work around by loading the
        # raw checkpoint and reconstructing the model manually — no .to() call.
        ckpt = torch.load(CKPT_PATH, map_location=torch.device("cpu"), weights_only=False)
        _MODEL = TemporalConvNet(**ckpt["hyper_parameters"])
        _MODEL.load_state_dict(ckpt["state_dict"])
        _MODEL.eval()
    return _MODEL


def predict_tcn(df: pd.DataFrame, start_dt: str, end_dt: str) -> pd.DataFrame:
    """Predict `sum_of_kWh` for rows in [start_dt, end_dt] where it is NaN.

    Returns: DataFrame with columns
        [eic_code, datetime, sum_of_kWh, dam_price, buy_bm_price, sell_bm_price]
    """
    print("Loading data…")
    df = inference_preprocessing_full(df, start_dt, end_dt)

    season_map = {
        12: "1", 1: "1", 2: "1",
        3: "2",  4: "2", 5: "2",
        6: "3",  7: "3", 8: "3",
        9: "4", 10: "4", 11: "4",
    }
    df["season"] = df["datetime"].dt.month.map(season_map)

    model = _ensure_loaded()

    # Recover the dataset definition (encoders, normalizer, column lists, …)
    # that was saved alongside the model weights at fit time.
    dataset_parameters = model.dataset_parameters

    # Cast categorical columns to str — same dtype handling as training.
    # Use `or []` not just a default: the checkpoint may store None explicitly.
    cat_cols = (
        list(dataset_parameters.get("static_categoricals") or [])
        + list(dataset_parameters.get("time_varying_known_categoricals") or [])
        + list(dataset_parameters.get("time_varying_unknown_categoricals") or [])
    )
    df = df.copy()
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str)

    # Rows we ultimately want predictions for: those with NaN target.
    missing_mask = df[Y_COL].isna()
    if not missing_mask.any():
        print("No rows with missing target — nothing to predict.")
        return df.iloc[0:0][
            [GROUP_COL, "datetime", Y_COL, "dam_price", "buy_bm_price", "sell_bm_price"]
        ]

    target_time_idxs = set(df.loc[missing_mask, TIME_IDX_COL].astype(int).tolist())
    start_idx = min(target_time_idxs)
    end_idx   = max(target_time_idxs)

    # Slide HORIZON-step windows over the prediction range, mirroring
    # `rolling_eval` from the training notebook. Each window: encoder context
    # is the rows up to `window_end - HORIZON`, decoder is the next HORIZON rows.
    records = []
    for w_start in range(start_idx, end_idx + 1, HORIZON):
        window_end = w_start + HORIZON - 1
        sub = df[df[TIME_IDX_COL] <= window_end].copy()
        if sub.empty:
            continue

        # TimeSeriesDataSet._check_tensors rejects ANY NaN in the target column
        # even with predict=True (the check is unconditional). Fill all NaN with
        # 0 as a structural placeholder — the model output replaces these values
        # and target_time_idxs already records which rows are the real forecast.
        if sub[Y_COL].isna().any():
            sub[Y_COL] = (
                sub.groupby(GROUP_COL)[Y_COL]
                .transform(lambda s: s.ffill().bfill().fillna(0))
            )

        ds = TimeSeriesDataSet.from_parameters(
            dataset_parameters,
            sub,
            predict=True,
            stop_randomization=True,
        )
        loader = ds.to_dataloader(
            train=False, batch_size=BATCH_SIZE, num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        result = model.predict(
            loader,
            return_index=True,
            trainer_kwargs={"accelerator": ACCELERATOR},
        )
        pred_np  = result[0].cpu().numpy()
        index_df = result[2]

        for i, (_, idx_row) in enumerate(index_df.iterrows()):
            base_t = int(idx_row[TIME_IDX_COL])
            for s in range(HORIZON):
                t = base_t + s
                if t not in target_time_idxs:
                    continue
                records.append({
                    GROUP_COL:    idx_row[GROUP_COL],
                    TIME_IDX_COL: t,
                    "pred":       float(pred_np[i, s]),
                })

    preds_df = pd.DataFrame(records)
    if preds_df.empty:
        print("No predictions produced (insufficient encoder history?).")
        return df.iloc[0:0][
            [GROUP_COL, "datetime", Y_COL, "dam_price", "buy_bm_price", "sell_bm_price"]
        ]

    # If a (group, t) appears in multiple windows, keep the earliest forecast
    # (longest lead time — same convention as day-ahead evaluation).
    preds_df = (
        preds_df.sort_values([GROUP_COL, TIME_IDX_COL])
                .drop_duplicates(subset=[GROUP_COL, TIME_IDX_COL], keep="first")
    )

    out = df.loc[
        missing_mask,
        [GROUP_COL, TIME_IDX_COL, "datetime", "dam_price", "buy_bm_price", "sell_bm_price"],
    ].merge(preds_df, on=[GROUP_COL, TIME_IDX_COL], how="left")

    out[Y_COL] = out["pred"].clip(lower=0)
    out = out.drop(columns=["pred", TIME_IDX_COL])

    return out[[GROUP_COL, "datetime", Y_COL,
                "dam_price", "buy_bm_price", "sell_bm_price"
                ]]


# ──────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_excel("../data/inference/one_station_sample.xlsx")
    df["Дата"] = pd.to_datetime(df["Дата"])

    df_preds = predict_tcn(df, start_dt="2025-09-01", end_dt="2025-10-30")
    print(df_preds)