# Electricity Forecasting — Interface

Streamlit UI that runs five forecasting models against historical electricity consumption data for EV charging stations.

---

## Available models

| Model | Checkpoint included | Notes |
|-------|--------------------|----|
| **XGBoost** | Yes (`models/xgb_model.json`) | Gradient boosting on engineered features |
| **Crossformer** | Yes (`models/crossformer_best.pt`) | Transformer-based, 168 h lookback, 48 h prediction step |
| **Chronos ZS** | N/A — zero-shot | Downloads `amazon/chronos-2` from HuggingFace on first run |
| **Chronos FT** | Request from author | Fine-tuned Chronos checkpoint; see note below |
| **TCN** | Request from author | Temporal Convolutional Network checkpoint; see note below |

> **Chronos FT and TCN checkpoints** are not included in this repository due to file size. They can be requested from the project author. Once obtained, place them at:
> - Chronos FT: `models/chronos-2-finetuned/` (directory containing `config.json` and `model.safetensors`)
> - TCN: `models/` (a `.ckpt` file — update the path constant in `tcn_inference.py` if the filename differs)

---

## Prerequisites

| Tool | Minimum version |
|------|----------------|
| Docker | 24.x |
| Docker Compose (plugin) | v2.x |

---

## Directory layout expected before you start

The container reads data from the project's `data/` directory, which is bind-mounted at `/data` inside the container. Ensure these files exist **on the host** before launching:

```
<project-root>/
├── data/
│   ├── additional_data/
│   │   ├── ВхідніДані.xlsx         # Station metadata (solar, max power, electricity prices)
│   │   └── EV_chargers.xlsx        # EV charger capacities and installation dates
│   ├── excel_files/
│   │   └── 2025_7-8.xlsx           # Static station dictionary (coordinates, oblast, DSO)
│   └── inference/
│       ├── one_station_sample.xlsx  # Single-station input example
│       └── all_stations_sample.xlsx # All-stations input example
└── interface/                       # ← run docker compose from here
    ├── Dockerfile
    ├── docker-compose.yml
    ├── requirements.txt
    ├── models/
    │   ├── crossformer_best.pt
    │   └── xgb_model.json
    └── *.py
```

The `data/inference/bronze/` directory is created automatically by the inference pipeline on first run.

---

## Input data format

Upload files must have the **same structure as `one_station_sample.xlsx`**:

| Column | Type | Description |
|--------|------|-------------|
| `EIC-код` | string | Station EIC identifier |
| `Дата` | datetime | Hourly timestamp |
| `Sum of кВт` | float | Hourly electricity consumption (kWh) |

Multi-station files follow the same column layout — see `all_stations_sample.xlsx`.

---

## Forecast window constraints

- **Lookback window**: the uploaded historical data must cover **at least 2 weeks** (336 hourly rows per station). Models use this history to produce forecasts.
- **Forecast start**: the forecast window always begins at the **first hour after the last timestamp** in the uploaded file. The start-date picker in the UI auto-fills to this value.
- **Forecast length**: there is no upper limit. For horizons longer than the model's native prediction step (48 h for Crossformer, 1 h for XGBoost), the model predicts **recursively** — each batch of predictions is fed back as history for the next batch.

---

## Performance notes

- **Single station**: typical inference completes in under a minute for all models.
- **All 395 stations** (`all_stations_sample.xlsx`): expect up to **30 minutes**. The main bottlenecks are weather data retrieval from the Open-Meteo API (one request per station) and iterative inference across 395 locations.
- All models run on **CPU only** (the Docker image installs the CPU-only PyTorch build to keep image size manageable).

---

## Build and run

All commands must be run from the **`interface/`** directory:

```bash
cd interface/

docker compose up --build -d
```

Then open **http://localhost:8501** in your browser.

---

## Stop and remove

```bash
docker compose down
```

To also remove the HuggingFace model cache volume (forces a fresh download of Chronos ZS weights next time):

```bash
docker compose down -v
```

---

## Important notes

### Chronos ZS — first-run download
The **Chronos ZS** model (`amazon/chronos-2`) is downloaded from HuggingFace the first time you run a forecast with it. The container must have outbound internet access. The weights are cached in a Docker named volume (`hf_cache`) so subsequent restarts reuse the local copy.

### CPU-only image
PyTorch is installed as a CPU-only build. All models run on CPU. Inference for Chronos or TCN over many stations can take several minutes.

### Weather stop signal
`weather.py` reads `weather_func_status.txt` to check for a stop signal during weather fetching. To halt a long-running weather fetch without killing the container:

```bash
docker exec electricity-forecasting bash -c "echo STOP > /app/weather_func_status.txt"
```

---

## View logs

```bash
docker compose logs -f
```
