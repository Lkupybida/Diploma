import pandas as pd

TRAIN_PATH_WITH_DATETME = "../data/gold/datetime_time_idx_cat/train.parquet"
VAL_PATH_WITH_DATETME   = "../data/gold/datetime_time_idx_cat/val.parquet"
TEST_PATH_WITH_DATETME  = "../data/gold/datetime_time_idx_cat/test.parquet"


TRAIN_PATH_OHE = "../data/gold/time_idx_ohe/train.parquet"
VAL_PATH_OHE   = "../data/gold/time_idx_ohe/val.parquet"
TEST_PATH_OHE  = "../data/gold/time_idx_ohe/test.parquet"

Y_COL        = 'sum_of_kWh'
GROUP_COL    = 'eic_code'
TIME_IDX_COL = "time_idx"

GLOBAL_MIN_DT = pd.Timestamp("2024-01-01 01:00:00", tz="Europe/Kyiv")

weather_cols_to_drop = [
    "apparent_temperature", "rain", "snowfall",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "surface_pressure", "wind_direction_10m", "wind_gusts_10m",
    "diffuse_radiation", "direct_normal_irradiance",
]

weather_cols_all = ['temperature_2m',
       'apparent_temperature', 'dew_point_2m', 'relative_humidity_2m',
       'precipitation', 'rain', 'snowfall', 'cloud_cover', 'cloud_cover_low',
       'cloud_cover_mid', 'cloud_cover_high', 'surface_pressure',
       'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m',
       'shortwave_radiation', 'diffuse_radiation', 'direct_normal_irradiance']

cat_columns = [
    'eic_code', 'osr_desc', 'station_type', 'oblast'
]

static_cols = [
    'latitude', 'longitude', 'eic_code', 'osr_desc', 'station_type', 'oblast'
]

calendar_cols = [
    'Month', 'Day', 'Hour', 'day_of_week', 'season'
]