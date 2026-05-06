# Electricity Forecasting — Docker Setup

Streamlit UI that runs five forecasting models (XGBoost, TCN, Crossformer,
Chronos FT, Chronos ZS) against historical electricity consumption data.

---

## Prerequisites

| Tool | Minimum version |
|------|----------------|
| Docker | 24.x |
| Docker Compose (plugin) | v2.x |

---

## Directory layout expected before you start

The container reads data from the project's `data/` directory, which is
bind-mounted read-only at `/data` inside the container.  Make sure these
files exist **on the host** before launching:

```
<project-root>/
├── data/
│   ├── additional_data/
│   │   ├── ВхідніДані.xlsx
│   │   └── EV_chargers.xlsx
│   ├── raw/
│   │   └── 8month2025.xlsx
│   └── inference/
│       └── bronze/
│           ├── no_weather.parquet
│           └── with_weather.parquet
└── interface/            ← run docker compose from here
    ├── Dockerfile
    ├── docker-compose.yml
    ├── requirements.txt
    ├── models/
    │   ├── chronos-2-finetuned/
    │   │   ├── config.json
    │   │   └── model.safetensors
    │   ├── cnn-epoch=00-val_loss=8.0403.ckpt
    │   ├── crossformer_best.pt
    │   └── xgb_model.json
    └── *.py
```

---

## Build and run

All commands must be run from the **`interface/`** directory (where
`docker-compose.yml` lives):

```bash
cd interface/

# Build the image and start the container in the foreground
docker compose up --build

# Or start detached (background):
docker compose up --build -d
```

Then open **http://localhost:8501** in your browser.

---

## Stop and remove

```bash
docker compose down
```

To also remove the HuggingFace model cache volume (forces a fresh download of
the Chronos ZS weights next time):

```bash
docker compose down -v
```

---

## Rebuild after code changes

If you edit any `.py` file or replace a model checkpoint, rebuild the image:

```bash
docker compose up --build
```

---

## Important notes

### Chronos ZS — first-run download
The **Chronos ZS** model (`amazon/chronos-2`) is downloaded from HuggingFace
the first time you run a forecast with it.  The container must have outbound
internet access for that download.  The weights are cached in a Docker named
volume (`hf_cache`) so subsequent restarts reuse the local copy.

### CPU-only image
PyTorch is installed as a CPU-only build to keep the image size manageable
(~4 GB smaller than the CUDA variant).  All models run on CPU.  Inference
for Chronos/TCN over many stations can take several minutes.

### Data is read-only
The `../data` bind-mount uses `:ro` (read-only).  The preprocessing pipeline
also writes intermediate parquet checkpoints to
`/data/inference/bronze/with_weather.parquet`.  If you want that write to
persist back to the host, remove the `:ro` flag from the volume line in
`docker-compose.yml`:

```yaml
- ../data:/data   # writable
```

### Weather status file
`weather.py` reads `weather_func_status.txt` in the working directory to
check for a stop signal during weather fetching.  The file is pre-created
empty inside the image (meaning "continue normally").  You can write `STOP`
into it to halt a long-running weather fetch without killing the container —
but only if you exec into the container first:

```bash
docker exec electricity-forecasting bash -c "echo STOP > /app/weather_func_status.txt"
```

---

## View logs

```bash
docker compose logs -f
```
