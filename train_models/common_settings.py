TRAIN_PATH = "../data/silver_money_calc/train.parquet"
VAL_PATH   = "../data/silver_money_calc/val.parquet"
TEST_PATH  = "../data/silver_money_calc/test.parquet"

Y_COL     = "sum_of_kWh"
GROUP_COL = "eic_code"

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
       'shortwave_radiation', 'diffuse_radiation', 'direct_normal_irradiance', 'daylight_duration']

cat_columns = [
    'eic_code', 'oblast', 'osr_desc'
]

static_cols = [
    'latitude', 'longitude'
]