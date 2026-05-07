# Data Pipeline

Scripts and notebooks that transform raw consumption Excel files into the
feature-engineered parquet files consumed by all training notebooks.

---

## Structure

```
data_pipeline/
├── data_procesing_pipeline.py      # Core processing functions — run this to rebuild gold data
├── data_processing_pipeline.ipynb  # End-to-end pipeline notebook, used for experiments
├── inference_data_preprocessing.py # Preprocessing variant used at inference time
├── weather.py                      # Open-Meteo API client with checkpoint/resume support
├── weather_func_status.txt         # Write "STOP" here to interrupt a running weather fetch
├── eda_v2.ipynb                    # Exploratory data analysis
└── add_air_raid_data.ipynb         # Tried to add air-raid alert features to the dataset
```

---

## Pipeline stages

`data_processing_pipeline.ipynb` runs the full pipeline end-to-end by calling
functions from `data_procesing_pipeline.py` in this order:

| Step | Function | Description |
|------|----------|-------------|
| 1 | `get_raw_consumption_data` | Concatenate all raw Excel files from `data/excel_files/` |
| 2 | `get_static_data_dict` | Load station coordinates, oblast, DSO, and type from the static dictionary |
| 3 | `add_solar` | Merge solar panel capacity per station (active after installation date) |
| 4 | `add_max_power` | Merge grid connection capacity per station |
| 5 | `drop_invalid_data` | Remove known bad data windows for specific stations |
| 6 | `keep_stations_with_90_days` | Drop stations with fewer than 90 days of observations |
| 7 | `add_weather_wrapper` | Fetch 18 hourly weather variables from Open-Meteo for each station's coordinates |
| 8 | `add_ev_power` | Merge cumulative EV charger capacity per station (active after installation date) |
| 9 | `add_calendar_features` | Add month, day, hour, day-of-week, and season columns |
| 10 | `build_time_index` | Add a global integer `time_idx` (hours since dataset start) |
| 11 | `trimm_stations` | For stations with gaps > 3 hours, keep only the longest uninterrupted tail |
| 12 | `add_prices` | Merge DAM, sell, and buy balancing-market prices |
| 13 | `fix_dst` | Reindex to a continuous hourly grid, interpolate single-hour DST gaps, drop stations with larger gaps |
| 14 | `label_test_val_train` | Assign `train` / `val` / `test` subsets by date |
| 15 | Export | Write two gold parquet variants (see below) |

---

## Gold output formats

The pipeline writes to `data/gold/`:

| Path | Contents | Used by |
|------|----------|---------|
| `gold/datetime_time_idx_global_cat/` | Keeps `datetime` column; categorical columns as strings | DL models (Crossformer, TCN, Chronos, LSTM, GRU, etc.) |
| `gold/time_idx_ohe/` | Drops `datetime`; categoricals one-hot encoded | ML models (XGBoost, LightGBM, etc.) and SARIMAX |

---

## Weather fetching

`weather.py` fetches historical hourly weather from the **Open-Meteo Archive API**
for each unique station coordinate. It writes incremental checkpoints so a
fetch interrupted halfway can resume without re-querying already-completed stations.

To stop a running fetch gracefully:

```bash
echo "STOP" > weather_func_status.txt
```

The 18 fetched weather variables are:

`temperature_2m`, `apparent_temperature`, `dew_point_2m`, `relative_humidity_2m`,
`precipitation`, `rain`, `snowfall`, `cloud_cover`, `cloud_cover_low`,
`cloud_cover_mid`, `cloud_cover_high`, `surface_pressure`, `wind_speed_10m`,
`wind_direction_10m`, `wind_gusts_10m`, `shortwave_radiation`, `diffuse_radiation`,
`direct_normal_irradiance`
