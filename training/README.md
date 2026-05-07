# Training

Training notebooks and shared Python utilities for all models.

## Structure

```
training/
├── common_settings.py     # Shared constants: paths, column names, feature lists, train/val split dates
├── data_loading.py        # Data loading and memory-optimization utilities
├── loss_funcs.py          # Custom loss functions (SMAPE, MAPE, RMSE, money and money_pct loss)
└── notebooks/
    ├── baseline.ipynb                       # Manual baseline, as used by OKKO
    ├── train_arimax.ipynb                   # ARIMAX
    ├── train_chronos.ipynb                  # Chronos (univariate)
    ├── train_chronos_multivariate.ipynb     # Chronos-2 multivariate variant
    ├── train_crossformer.ipynb              # Crossformer v1
    ├── train_crossformer_v2.ipynb           # Crossformer v2
    ├── train_crossformer_v3.ipynb           # Crossformer v3 — best checkpoint
    ├── train_gru.ipynb                      # GRU
    ├── train_lgbm.ipynb                     # LightGBM
    ├── train_lstm.ipynb / v0 / v2 / v3      # LSTM variants
    ├── train_moirai.ipynb / v2              # Moirai (Salesforce)
    ├── train_moirai_multivariate.ipynb      # Moirai multivariate variant
    ├── train_nhits.ipynb                    # N-HiTS
    ├── train_patchtst.ipynb / v2            # PatchTST
    ├── train_rose.ipynb / v2 / v3           # RoSE
    ├── train_sarimax.ipynb                  # SARIMAX
    ├── train_serimax.ipynb                  # typo, second iteration of SARIMAX
    ├── train_tcn.ipynb                      # TCN — production checkpoint
    ├── train_tft.ipynb                      # Temporal Fusion Transformer
    ├── train_xgb.ipynb                      # XGBoost — production checkpoint
    ├── outages.ipynb                        # Outage / anomaly analysis
    └── old_notebooks/                       # Archived early iterations
```

## Setup

Install dependencies from the root:

```bash
pip install -r ../requirements.txt
```

## Data

All notebooks load from `../data/gold/` parquet files. Paths and column names are
configured in `common_settings.py`. Generate the gold data via the full preprocessing
pipeline before running any training notebook.

## Experiment tracking

MLflow runs are logged to `notebooks/mlruns/`. To view the tracking UI:

```bash
cd notebooks/
mlflow ui
```

Then open **http://localhost:5000** in your browser.
