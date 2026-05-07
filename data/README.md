# Data

All data files used for training and inference.

## Structure

```
data/
├── additional_data/
│   ├── ВхідніДані.xlsx           # Station metadata: solar capacity, max power, electricity prices
│   └── EV_chargers.xlsx          # EV charger capacity and installation dates per station
├── excel_files/
│   └── 2025_7-8.xlsx             # Static station dictionary (coordinates, oblast, DSO info)
├── gold/                          # Pre-processed parquet files ready for model training
│   ├── datetime_time_idx_global_cat/
│   │   ├── train.parquet
│   │   ├── val.parquet
│   │   └── test.parquet
│   └── time_idx_ohe/              # Same split with one-hot-encoded categorical features
│       ├── train.parquet
│       ├── val.parquet
│       └── test.parquet
└── inference/
    ├── one_station_sample.xlsx    # Sample input for interface — single station (use this)
    ├── all_stations_sample.xlsx   # Sample input interface — all 395 stations
    └── bronze/                    # Created at runtime by the inference pipeline
        ├── no_weather.parquet     # Preprocessed data before weather enrichment
        └── with_weather.parquet   # Preprocessed data after weather enrichment (cached)
```

## Notes

- The `bronze/` directory is used for checkpoints when adding weather 
- The `gold/` parquet files are the output of the full preprocessing pipeline and are the direct inputs to all training notebooks.
- The `gold/datetime_time_idx_global_cat/` parquet files are used for most models and have string dtype columns
- The `gold/time_idx_ohe/` parquet files are used for models that do not support categorical features and have one-hot encoded columns
- Train/val/test split: **train** ends 2025-07-01, **val** ends 2025-08-01, **test** is everything after.

## Data description

Each row represents one hour of observations for a single charging station. Columns are grouped below by category.

### Identifiers and target

| Column | Type | Description |
|---|---|---|
| `datetime` | int64 (ns) | Timestamp of the observation in nanoseconds since epoch (hourly resolution). |
| `eic_code` | string | Unique station identifier (Energy Identification Code). |
| `sum_of_kWh` | float | **Target variable.** Total energy consumed at the station during the hour, in kWh. |

### Station metadata (static per station)

| Column | Type | Description |
|---|---|---|
| `max_power` | float | Maximum grid connection capacity of the station, in kW. |
| `max_solar` | float | Installed solar PV capacity at the station, in kW. |
| `max_ev` | float | Installed EV charger capacity at the station, in kW (0 if no chargers). |
| `latitude` | float | Station latitude in decimal degrees (WGS84). |
| `longitude` | float | Station longitude in decimal degrees (WGS84). |
| `dso_desc` | string | Distribution System Operator (DSO) servicing the station. |
| `station_type` | string | Station category (e.g. `ОККО-комплекс`). |
| `oblast` | string | Ukrainian oblast (administrative region) where the station is located. |

### Weather features (from Open-Meteo, hourly)

| Column | Unit | Description |
|---|---|---|
| `temperature_2m` | °C | Air temperature at 2 m above ground. |
| `relative_humidity_2m` | % | Relative humidity at 2 m. |
| `dew_point_2m` | °C | Dew point at 2 m. |
| `apparent_temperature` | °C | Perceived ("feels like") temperature. |
| `surface_pressure` | hPa | Atmospheric pressure at the surface. |
| `cloud_cover` | % | Total cloud cover. |
| `cloud_cover_low` | % | Low-altitude cloud cover (< 3 km). |
| `cloud_cover_mid` | % | Mid-altitude cloud cover (3–8 km). |
| `cloud_cover_high` | % | High-altitude cloud cover (> 8 km). |
| `wind_speed_10m` | km/h | Wind speed at 10 m. |
| `wind_direction_10m` | ° | Wind direction at 10 m (0–360°, meteorological convention). |
| `wind_gusts_10m` | km/h | Wind gust speed at 10 m. |
| `shortwave_radiation` | W/m² | Global horizontal irradiance (total solar radiation). |
| `direct_normal_irradiance` | W/m² | Direct beam solar radiation perpendicular to the sun. |
| `diffuse_radiation` | W/m² | Diffuse (scattered) solar radiation on a horizontal plane. |
| `precipitation` | mm | Total precipitation (rain + snow water-equivalent). |
| `rain` | mm | Liquid precipitation only. |
| `snowfall` | cm | Snowfall amount. |

### Calendar and time features

| Column | Type | Description |
|---|---|---|
| `Month` | int (1–12) | Calendar month. |
| `Day` | int (1–31) | Day of the month. |
| `Hour` | int (0–23) | Hour of the day. |
| `day_of_week` | int (0–6) | Day of week (0 = Monday, 6 = Sunday). |
| `season` | string | Season indicator (winter, spring, summer, autumn). |
| `time_idx` | int | Monotonic hourly index used by the forecasting models for sequence ordering. |

### Electricity market prices (UAH/MWh)

| Column | Description |
|---|---|
| `sell_bm_price` | Balancing market price for selling energy back to the grid. |
| `buy_bm_price` | Balancing market price for buying energy from the grid. |
| `dam_price` | Day-Ahead Market clearing price. |

### Sample row

```
datetime,eic_code,sum_of_kWh,max_power,max_solar,max_ev,latitude,longitude,dso_desc,station_type,oblast,temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,surface_pressure,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,wind_speed_10m,wind_direction_10m,wind_gusts_10m,shortwave_radiation,direct_normal_irradiance,diffuse_radiation,precipitation,rain,snowfall,Month,Day,Hour,day_of_week,season,time_idx,sell_bm_price,buy_bm_price,dam_price
1753995600000000000,62Z0008583037334,22,110,25.52,0,48.44243,22.19219,Ужгород,ОККО-комплекс,Закарпатська,17.1,92,15.9,18.5,1001.4,3,0,3,0,4.4,348,7.6,0,0,0,0,0,0,8,1,0,4,3,13870,4.55,5880,5600
```