import pandas as pd
import numpy as np
import os
from tqdm.auto import tqdm
from weather import smart_requests

def get_raw_consumption_data(folder_path):

    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]

    df = pd.DataFrame()
    for f in tqdm(file_paths):
        df = pd.concat([df, pd.read_excel(f, sheet_name="DATA", header=5)], ignore_index=True)

    df = df[~df['EIC-код'].isin(['Grand Total', 'Загальний підсумок'])].copy()

    return df

def get_static_data_dict(path):

    static_data_dict = pd.read_excel(path, sheet_name="Dict")

    eic_code_to_missing_coords = {
        "62Z0602349894257": [48.763904006776954, 28.077456990197557],
        "62Z6035944504754": [49.153510259625534, 23.049801982501872],
        "62Z8888861892130": [48.309249416757545, 25.11895645420284],
        "62Z1209368438999": [49.78531388363913, 24.927028292321978],
        "62Z1771663109656": [49.453445499775405, 23.020911411362146],
        "62Z2532379651825": [49.29362429838761, 23.41030049170128],
        "62Z3261331410025": [49.56421176098104, 23.304357225434188],
        "62Z3627036877597": [48.51882104530565, 28.760435421364264],
        "62Z5422632680702": [50.22822399875523, 24.83726310970472],
        "62Z0763594350330": [49.369735771908296, 23.47527668624802],
        "62Z7029546479880": [49.77948499692157, 24.013408048459073],
        "62Z7072263538637": [49.86242688799433, 24.062926768794956],
        "62Z7766151236986": [50.46762760936502, 24.26123769025777],
        "62Z5064314448909": [48.149046045899055, 23.322957181272095],
        "62Z9342951787758": [49.28552561066835, 23.534192240667057],
        "62Z8521105012164": [48.36054435942602, 29.536275478873062],
        "62Z1142387881356": [49.76402754584019, 31.39902598856454],
        "62Z6624735969589": [48.94419405940688, 24.149295411716132],
        "62Z5956494916290": [49.95178196130912, 25.31699531547816],
        "62Z0717322121288": [48.207039717684104, 23.522199099176],
        "62Z568600499251K": [49.74638889010666, 31.486251700927244],
    }

    lat_map = {k: v[0] for k, v in eic_code_to_missing_coords.items()}
    lon_map = {k: v[1] for k, v in eic_code_to_missing_coords.items()}

    static_data_dict["GPS-координати - Широта"] = (
        static_data_dict["GPS-координати - Широта"]
        .replace(0, pd.NA)
        .fillna(static_data_dict["EIC-код"].map(lat_map))
    )

    static_data_dict["GPS-координати - Довгота"] = (
        static_data_dict["GPS-координати - Довгота"]
        .replace(0, pd.NA)
        .fillna(static_data_dict["EIC-код"].map(lon_map))
    )

    return static_data_dict

def add_solar(df, path = "data/additional_data/ВхідніДані.xlsx"):

    solar_dict = pd.read_excel(path, sheet_name="СЕС", header=0)
    solar_dict = solar_dict[solar_dict['Plant Name'] != 'Total'].copy()
    cols_to_rename = {
        'Unicod': 'Унікод',
        'Grid Connection Date': 'solar_install_date',
        'Max Power, kWt': 'max_solar',
    }
    solar_dict = solar_dict.rename(columns=cols_to_rename).copy()
    solar_dict['Унікод'] = solar_dict['Унікод'].astype(int)
    solar_dict = solar_dict[['Унікод', 'solar_install_date', 'max_solar']].copy()

    df = df.merge(solar_dict, on='Унікод', how='left', suffixes=('', '_dict'))

    mask = (df['Дата'] <= df['solar_install_date']) | (df['solar_install_date'].isna())
    df.loc[mask, 'max_solar'] = 0

    return df

def add_max_power(df, path = "data/additional_data/ВхідніДані.xlsx"):

    max_power_dict = pd.read_excel(path, sheet_name="Потужність", header=2)
    cols_to_rename = {
        'Унікод АЗС': 'Унікод',
        'Існуюча потужність, кВт': 'max_power'
    }
    max_power_dict = max_power_dict.rename(columns=cols_to_rename).copy()
    max_power_dict = max_power_dict[['Унікод', 'max_power']].copy()

    df = df.merge(max_power_dict, on='Унікод', how='left', suffixes=('', '_dict'))

    mask = df['max_power'].isna()
    df.loc[mask, 'max_power'] = (
        df.groupby('EIC-код')['Sum of кВт'].transform('max')
    )

    return df

def drop_invalid_data(df):
    # --- 62Z3584090129468: drop data before 2024-10-01 ---
    mask = (df['EIC-код'] == '62Z3584090129468') & (df['Дата'] < '2024-10-02')
    df = df[~mask]

    # --- 62Z0101426517156: cap 'Sum of кВт' at 210 ---
    mask = df['EIC-код'] == '62Z0101426517156'
    df.loc[mask & (df['Sum of кВт'] > 210), 'Sum of кВт'] = 210

    mask = (df['EIC-код'] == '62Z0101426517156') & (df['Дата'] < '2025-01-01')
    df = df[~mask]

    # --- 62Z4869256917783: drop before 2025, scale 2025-05-07 09:00–2025-05-31 23:00 ---
    mask_eic = df['EIC-код'] == '62Z4869256917783'
    # Drop before 2025
    df = df[~(mask_eic & (df['Дата'] < '2025-06-01'))]

    # --- 62Z2655661084314: scale 2025-04-01 09:00–2025-04-30 22:00 to mean of Mar+May 2025 ---
    mask_eic = df['EIC-код'] == '62Z2655661084314'

    df = df[~(mask_eic & (df['Дата'] < '2025-05-01'))]

    # --- 62Z4948939192262: drop before 2024-05-05 ---
    mask = (df['EIC-код'] == '62Z4948939192262') & (df['Дата'] < '2025-05-05')
    df = df[~mask]

    # --- 62Z5644272506149: drop before 2025-05-05 ---
    mask = (df['EIC-код'] == '62Z5644272506149') & (df['Дата'] < '2025-05-05')
    df = df[~mask]

    # --- 62Z9512175266041: drop before 2025-05-05 ---
    mask = (df['EIC-код'] == '62Z9512175266041') & (df['Дата'] < '2025-05-05')
    df = df[~mask]

    # --- 62Z2986251549800: drop before 2024-05-05 ---
    mask = (df['EIC-код'] == '62Z2986251549800') & (df['Дата'] < '2024-05-05')
    df = df[~mask]

    # --- 62Z7878159195687: drop before 2024-11-01 ---
    mask = (df['EIC-код'] == '62Z7878159195687') & (df['Дата'] < '2024-11-01')
    df = df[~mask]

    mask = (
            (df["EIC-код"] == "62Z8521105012164") &
            (df["Дата"] < "2024-06-01")
    )

    df = df.loc[~mask]

    return df

def keep_stations_with_90_days(df):
    print(f"Stations before dropping: {len(df['eic_code'].unique())}, rows: {len(df)}")

    global_end = df["datetime"].max()

    # station-level stats
    g = df.groupby("eic_code")["datetime"]

    station_stats = pd.DataFrame({
        "start": g.min(),
        "end": g.max(),
        "n_hours": g.count()
    })

    # rule 1: must have data near dataset end (within 7 days tolerance)
    station_stats["reaches_end"] = station_stats["end"] >= (global_end - pd.Timedelta(days=7))

    # rule 2: must have at least 3 months of REAL observations
    station_stats[">=3_months_data"] = station_stats["n_hours"] >= 90 * 24

    # final filter
    valid_stations = station_stats[
        station_stats["reaches_end"] &
        station_stats[">=3_months_data"]
        ].index

    df = df[df["eic_code"].isin(valid_stations)].copy()

    print(f"Stations after dropping: {len(df['eic_code'].unique())}, rows: {len(df)}")
    return df

def add_weather_wrapper(df, no_weather_path, weather_path, checkpoint_path, weather_cols):

    df.to_parquet(no_weather_path, index=False)
    df.to_parquet(checkpoint_path, index=False)

    df_with_weather = smart_requests(
        path_in=checkpoint_path,
        path_out=weather_path,
        checkpoint_path=checkpoint_path,
        eic_code_name='eic_code',
        lat_col='latitude',
        lon_col='longitude',
        weather_vars=weather_cols,
    )

    return df_with_weather

def add_ev_power(df, static_path = "data/raw/8month2025.xlsx", ev_path = "data/additional_data/EV_chargers.xlsx"):

    static_data_dict = pd.read_excel(static_path, sheet_name="Dict")[['EIC-код', 'Унікод']]

    ev_data = pd.read_excel(ev_path)[
        ['Унікод АЗС', 'Дата запуску', 'Потужність станції, кВт']]
    cols_to_rename = {
        'Унікод АЗС': 'Унікод',
        'Дата запуску': 'ev_start',
        'Потужність станції, кВт': 'ev_power'
    }
    ev_data = ev_data.rename(columns=cols_to_rename).copy()

    ev_data = ev_data.merge(static_data_dict, on='Унікод', how='left', suffixes=('', '_dict'))
    ev_data.drop(columns=['Унікод'], inplace=True)
    ev_data.dropna(inplace=True)
    ev_data = ev_data.rename(columns={"EIC-код": "eic_code"})
    ev_data = ev_data.sort_values(["eic_code", "ev_start"])

    ev_data["ev_start_utc"] = ev_data["ev_start"].dt.tz_localize("UTC")

    merged = df.merge(
        ev_data[["eic_code", "ev_start_utc", "ev_power"]],
        on="eic_code",
        how="left",
    )

    mask_installed = (
            merged["ev_start_utc"].isna() |
            (merged["ev_start_utc"] <= merged["datetime"])
    )
    merged = merged[mask_installed]

    max_ev_power = (
        merged
        .groupby(["eic_code", "datetime"], sort=False)["ev_power"]
        .sum()
        .reset_index()
        .rename(columns={"ev_power": "max_ev_power_kW"})
    )

    df_result = df.merge(max_ev_power, on=["eic_code", "datetime"], how="left")
    df_result["max_ev_power_kW"] = df_result["max_ev_power_kW"].fillna(0)

    return df_result

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["datetime"]        = pd.to_datetime(df["datetime"])
    df["Month"]       = df["datetime"].dt.month.astype(str)
    df["Day"]         = df["datetime"].dt.day.astype(str)
    df["Hour"]        = df["datetime"].dt.hour.astype(str)
    df["day_of_week"] = df["datetime"].dt.dayofweek.astype(str)
    season_map = {
        12: "winter", 1: "winter", 2: "winter",
        3: "spring",  4: "spring", 5: "spring",
        6: "summer",  7: "summer", 8: "summer",
        9: "autumn", 10: "autumn", 11: "autumn",
    }
    df["season"] = df["datetime"].dt.month.map(season_map)
    return df

GLOBAL_MIN_DT = None

def build_time_index(df: pd.DataFrame) -> pd.DataFrame:
    global GLOBAL_MIN_DT
    # df["datetime"] = pd.to_datetime(df["datetime"])
    # if GLOBAL_MIN_DT is None:
    #     GLOBAL_MIN_DT = df["datetime"].min()
    # df['datetime'] = df['datetime'].dt.tz_convert('UTC')

    df = df.sort_values(['eic_code', 'datetime'])

    df['time_idx'] = (
        (df['datetime'] - df['datetime'].min())
        .dt.total_seconds() // 3600
    ).astype(int)
    return df

def find_missing_timesteps(df):
    rows = []
    for grp, gdf in df.groupby("eic_code"):
        full_range = set(range(gdf["time_idx"].min(), gdf["time_idx"].max() + 1))
        present = set(gdf["time_idx"].values)
        missing = sorted(full_range - present)
        if missing:
            rows.append({"eic_code": grp, "missing_count": len(missing), "missing_time_idx": missing})
    if not rows:
        return pd.DataFrame(columns=["eic_code", "missing_count", "missing_time_idx"])
    return pd.DataFrame(rows).sort_values("missing_count", ascending=False).reset_index(drop=True)

def trimm_stations(df):

    missing = find_missing_timesteps(df)

    def get_longest_uninterrupted_from_end(gdf, min_days = 90):
        gdf = gdf.sort_values("datetime").reset_index(drop=True)
        deltas = gdf["datetime"].diff()
        gap_positions = deltas[deltas > pd.Timedelta("1h")].index.tolist()
        last_gap = gap_positions[-1] if gap_positions else 0
        clean_block = gdf.iloc[last_gap:]
        if (clean_block["datetime"].max() - clean_block["datetime"].min()).days < min_days:
            return None
        return clean_block

    significantly_missing = missing[missing["missing_count"] > 3]["eic_code"].tolist()

    kept, dropped = [], []
    for eic in significantly_missing:
        result = get_longest_uninterrupted_from_end(df[df["eic_code"] == eic])
        if result is None:
            dropped.append(eic)
        else:
            kept.append(result)

    clean_others = df[~df["eic_code"].isin(significantly_missing)]
    df = pd.concat([clean_others] + kept, ignore_index=True).sort_values(
        ["eic_code", "datetime"]
    ).reset_index(drop=True)
    df = df[~df["eic_code"].isin(dropped)].reset_index(drop=True)

    print(f"Trimmed : {len(significantly_missing)} stations")
    print(f"Kept    : {len(kept)}")
    print(f"Dropped : {len(dropped)}")
    print(f"Remaining stations : {df['eic_code'].nunique()}")
    print(f"Remaining rows     : {len(df):,}")

    return df

def add_prices(df, prices_path = 'data/additional_data/ВхідніДані.xlsx'):

    prices_df = pd.read_excel(prices_path, sheet_name="Prices", header=1)
    cols_to_rename = {
        'Row Labels': 'datetime',
        'БР- (продаж)': 'sell_bm_price',
        'БР+ (купівля)': 'buy_bm_price',
        'РДН': 'dam_price'
    }
    prices_df = prices_df.rename(columns=cols_to_rename).copy()

    is_datetime = prices_df['datetime'].astype(str).str.contains('-')

    prices_df.loc[is_datetime, 'base_date'] = pd.to_datetime(prices_df.loc[is_datetime, 'datetime'])

    # forward fill the date
    prices_df['base_date'] = prices_df['base_date'].ffill()

    prices_df['hour'] = pd.to_numeric(prices_df['datetime'], errors='coerce')

    prices_df['datetime'] = prices_df['base_date']

    mask = prices_df['hour'].notna()
    prices_df.loc[mask, 'datetime'] = (
            prices_df.loc[mask, 'base_date'] +
            pd.to_timedelta(prices_df.loc[mask, 'hour'] - 1, unit='h')
    )
    prices_df['datetime'] = prices_df['datetime'].dt.tz_localize(
        "Europe/Kyiv",
        ambiguous="NaT"
    )

    prices_df = prices_df.drop(columns=['base_date', 'hour'])
    prices_df.dropna(inplace=True)
    df = df.merge(prices_df, on='datetime', how='left', suffixes=('', '_dict'))

    return df

def fix_dst(df, static_cols, calendar_cols):
    num_cols = df.select_dtypes(include="number").columns.tolist()
    interp_cols = [c for c in num_cols if c not in calendar_cols + ["time_idx"]]

    fixed = []
    dropped = []

    for grp, gdf in df.groupby("eic_code"):
        gdf = gdf.copy().sort_values("datetime")

        gdf = gdf.drop_duplicates(subset="datetime", keep="first").sort_values("datetime")

        tz = gdf["datetime"].dt.tz
        full_idx = pd.date_range(
            gdf["datetime"].min(),
            gdf["datetime"].max(),
            freq="h",
            tz=tz
        )

        missing_mask = ~full_idx.isin(gdf["datetime"])
        if missing_mask.any():
            runs = []
            run_len = 0
            for m in missing_mask:
                if m:
                    run_len += 1
                else:
                    if run_len > 0:
                        runs.append(run_len)
                    run_len = 0
            if run_len > 0:
                runs.append(run_len)

            max_consecutive = max(runs)
            n_missing = missing_mask.sum()

            if max_consecutive > 1:
                print(f"  DROP  {grp}: {n_missing} missing timestamps, "
                      f"max consecutive gap = {max_consecutive}h")
                dropped.append(grp)
                continue
            else:
                print(f"  FILL  {grp}: {n_missing} single-hour gap(s) — interpolating")

        gdf = gdf.set_index("datetime").reindex(full_idx)
        gdf.index.name = "datetime"

        for col in static_cols:
            gdf[col] = gdf[col].ffill().bfill()

        gdf[interp_cols] = gdf[interp_cols].interpolate(method="time")

        gdf["Month"]       = gdf.index.month
        gdf["Day"]         = gdf.index.day
        gdf["Hour"]        = gdf.index.hour
        gdf["day_of_week"] = gdf.index.dayofweek
        gdf["season"]      = (gdf.index.month % 12 // 3 + 1).astype(str)

        gdf = gdf.reset_index()
        fixed.append(gdf)

    print(f"\nSummary: kept {len(fixed)} stations, dropped {len(dropped)} stations")
    if dropped:
        print(f"Dropped: {dropped}")

    result = pd.concat(fixed, ignore_index=True)
    result = result.sort_values(["datetime", "eic_code"]).reset_index(drop=True)

    base = result.groupby("eic_code")["datetime"].transform(
        lambda s: (s - s.min()).dt.total_seconds() // 3600
    ).astype(int)

    result["time_idx"] = base

    cols_to_ffill = ['dam_price', 'buy_bm_price', 'sell_bm_price']

    result[cols_to_ffill] = result[cols_to_ffill].ffill()

    return result

def label_test_val_train(df, train_end = "2025-07-01", val_end = "2025-08-01"):
    train_end = pd.Timestamp(train_end, tz=df["datetime"].dt.tz)
    val_end = pd.Timestamp(val_end, tz=df["datetime"].dt.tz)

    df["data_subset"] = "test"

    df.loc[df["datetime"] < train_end, "data_subset"] = "train"

    df.loc[
        (df["datetime"] >= train_end) &
        (df["datetime"] < val_end),
        "data_subset"
    ] = "val"

    return df

weather_cols_to_create_all = ['temperature_2m',
                              'apparent_temperature', 'dew_point_2m', 'relative_humidity_2m',
                              'precipitation', 'rain', 'snowfall', 'cloud_cover', 'cloud_cover_low',
                              'cloud_cover_mid', 'cloud_cover_high', 'surface_pressure',
                              'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m',
                              'shortwave_radiation', 'diffuse_radiation', 'direct_normal_irradiance']

no_weather_path = "../data/test/bronze/no_weather.parquet"
weather_path = "../data/test/bronze/with_weather.parquet"
checkpoint_path = "../data/test/bronze/checkpoint.parquet"

silver_path = "../data/test/silver/feature_engineered_built_idx_split.parquet"

gold_folder = "data/test/gold"

cols_to_ohe = [
    'dso_desc', 'station_type', 'oblast',
    'Month', 'Day', 'Hour', 'day_of_week', 'season'
]

def process_all_data():

    df = get_raw_consumption_data(folder_path="../data/excel_files")

    df = df[df['EIC-код'] == df['EIC-код'][0]].copy()

    static_path = "../data/raw/8month2025.xlsx"
    static_data_dict = get_static_data_dict(path=static_path)

    df = df.merge(static_data_dict, on='EIC-код', how='left', suffixes=('', '_dict'))

    df = add_solar(df)

    df = add_max_power(df)

    df['Дата'] = pd.to_datetime(df['Дата'], errors='coerce')
    df['Дата'] = df['Дата'] + pd.to_timedelta(df['Hour'].fillna(0), unit='h')

    df = drop_invalid_data(df)

    cols_to_leave = ['EIC-код',
                     'Дата', 'Sum of кВт',
                     'Тип',
                     'Область', 'GPS-координати - Широта', 'GPS-координати - Довгота', 'ОСР опис', 'Адреса',
                     'max_power', 'max_solar']

    rename_cols = {
        'Дата': 'datetime',
        'Sum of кВт': 'sum_of_kWh',
        'EIC-код': 'eic_code',
        'Тип': 'station_type',
        'Область': 'oblast',
        'GPS-координати - Широта': 'latitude',
        'GPS-координати - Довгота': 'longitude',
        'ОСР опис': 'osr_desc',
        'Адреса': 'address'
    }

    df = df[cols_to_leave].copy()
    df = df.rename(columns=rename_cols)

    df = keep_stations_with_90_days(df)

    df = add_weather_wrapper(df, no_weather_path, weather_path, checkpoint_path, weather_cols_to_create_all)

    df['datetime'] = pd.to_datetime(df['datetime'])

    df = add_ev_power(df, static_path, "../data/additional_data/EV_chargers.xlsx")

    df = add_calendar_features(df)

    df = build_time_index(df)

    df = df[[
        'eic_code', 'datetime', 'sum_of_kWh', 'time_idx',
        'max_power', 'max_solar', 'max_ev_power_kW',
        'latitude', 'longitude', 'osr_desc', 'station_type', 'oblast']
        +
        weather_cols_to_create_all
    ]

    cols_to_rename = {
        'max_power': 'max_power',
        'max_solar': 'max_solar',
        'max_ev_power_kW': 'max_ev',
        'osr_desc': 'dso_desc',
    }

    df = df.rename(columns=cols_to_rename)

    df = trimm_stations(df)

    df = add_prices(df)

    static_cols = [
        'latitude', 'longitude', 'eic_code', 'dso_desc', 'station_type', 'oblast'
    ]

    calendar_cols = [
        'Month', 'Day', 'Hour', 'day_of_week', 'season'
    ]

    df = fix_dst(df, static_cols, calendar_cols)

    print(f"Number of missing timesteps: {len(find_missing_timesteps(df))}")

    df = label_test_val_train(df)

    df.to_parquet(silver_path, index=False)

    GLOBAL_MIN_DT = df["datetime"].min()

    print(GLOBAL_MIN_DT)

    df = df.sort_values(['eic_code', 'datetime'])

    df['time_idx'] = (
            (df['datetime'] - GLOBAL_MIN_DT)
            .dt.total_seconds() // 3600
    ).astype(int)

    # for DL models & transformers

    df[df['data_subset'] == "train"].drop(columns=["data_subset"]).to_parquet(
        gold_folder + "/datetime_time_idx_global_cat/train.parquet", index=False)
    df[df['data_subset'] == "val"].drop(columns=["data_subset"]).to_parquet(
        gold_folder + "/datetime_time_idx_global_cat/val.parquet", index=False)
    df[df['data_subset'] == "test"].drop(columns=["data_subset"]).to_parquet(
        gold_folder + "/datetime_time_idx_global_cat/test.parquet", index=False)

    df_ohe = pd.get_dummies(
        df,
        columns=cols_to_ohe,
        drop_first=False
    )

    # for ML models & SARIMAX

    df_ohe[df_ohe['data_subset'] == "train"].drop(columns=["data_subset", "datetime"]).to_parquet(
        gold_folder + "/time_idx_ohe/train.parquet", index=False)
    print("Train:", len(df_ohe[df_ohe['data_subset'] == "train"]))
    df_ohe[df_ohe['data_subset'] == "val"].drop(columns=["data_subset", "datetime"]).to_parquet(
        gold_folder + "/time_idx_ohe/val.parquet", index=False)
    print("Val:", len(df_ohe[df_ohe['data_subset'] == "val"]))
    df_ohe[df_ohe['data_subset'] == "test"].drop(columns=["data_subset", "datetime"]).to_parquet(
        gold_folder + "/time_idx_ohe/test.parquet", index=False)
    print("Test:", len(df_ohe[df_ohe['data_subset'] == "test"]))