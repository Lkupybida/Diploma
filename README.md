# Electricity Consumption Forecasting for EV Charging Stations

Diploma project: multi-model hourly electricity consumption forecasting for EV charging stations across Ukraine (395 stations).

## Repository structure

```
.
├── data/               # Raw, processed, and inference data
├── data_pipeline/      # Scripts and notebooks that build the gold training data
├── interface/          # Streamlit forecasting UI (Docker-based)
├── training/           # Training notebooks and shared utilities
└── requirements.txt    # Training environment dependencies
```

## Quick start — forecasting UI

The interface runs inside Docker and requires no local Python setup:

```bash
cd interface/
docker compose up --build
```

Open **http://localhost:8501** in your browser. See [`interface/README.md`](interface/README.md) for full setup instructions, model availability, and input data requirements.

## Data pipeline

Raw Excel files → gold parquet files ready for training. See [`data_pipeline/README.md`](data_pipeline/README.md) for the full stage-by-stage breakdown.

## Training

Training notebooks are in `training/notebooks/`. Experiments are tracked with MLflow (logs in `training/notebooks/mlruns/`). See [`training/README.md`](training/README.md) for the full notebook index and setup instructions.

## Data

Sample inference files are in `data/inference/`. Gold (feature-engineered) parquet files used for training are in `data/gold/`. See [`data/README.md`](data/README.md) for the full layout and data description.

## Training dependencies

```bash
pip install -r requirements.txt
```

The interface has its own isolated `interface/requirements.txt` managed via Docker.
