from pathlib import Path
import time
import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm

def add_openmeteo_weather_sync(
    df: pd.DataFrame,
    weather_vars: list[str],
    datetime_col: str = "datetime",
    lat_col: str = "GPS-координати - Широта",
    lon_col: str = "GPS-координати - Довгота",
    timezone: str = "Europe/Kyiv",
    request_timeout: int = 120,
) -> pd.DataFrame:
    """
    Synchronously fetch Open-Meteo archive weather for each unique (lat, lon)
    over its full datetime range, then merge the weather back into the same df.

    Returns a new dataframe with added weather columns.

    Assumptions:
    - df has one row per observation with datetime + lat + lon
    - weather is hourly
    - datetime should be interpreted in Europe/Kyiv local time
    """

    if not weather_vars:
        raise ValueError("weather_vars must be a non-empty list")

    required = {datetime_col, lat_col, lon_col}
    missing_required = required - set(df.columns)
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    out_df = df.copy()

    def to_kyiv_tz(series: pd.Series) -> pd.Series:
        series = pd.to_datetime(series, errors="coerce")
        if series.dt.tz is None:
            try:
                return series.dt.tz_localize(
                    "Europe/Kyiv",
                    ambiguous="infer",
                    nonexistent="shift_forward",
                )
            except Exception:
                return series.dt.tz_localize(
                    "Europe/Kyiv",
                    ambiguous=False,
                    nonexistent="shift_forward",
                )
        return series.dt.tz_convert("Europe/Kyiv")

    def floor_hour_safe(series: pd.Series) -> pd.Series:
        return series.dt.floor(
            "h",
            ambiguous=False,
            nonexistent="shift_forward",
        )

    out_df[datetime_col] = to_kyiv_tz(out_df[datetime_col])
    if out_df[datetime_col].isna().any():
        raise ValueError(f"Some values in {datetime_col} could not be parsed as datetime.")

    out_df[datetime_col] = floor_hour_safe(out_df[datetime_col])

    for col in weather_vars:
        if col not in out_df.columns:
            out_df[col] = np.nan

    valid_mask = out_df[lat_col].notna() & out_df[lon_col].notna()
    req_df = out_df.loc[valid_mask, [datetime_col, lat_col, lon_col]].copy()

    if req_df.empty:
        return out_df

    grouped = req_df.groupby([lat_col, lon_col], dropna=False)

    weather_frames = []
    url = "https://archive-api.open-meteo.com/v1/archive"

    for (lat, lon), g in grouped:
        start_dt = g[datetime_col].min()
        end_dt = g[datetime_col].max()

        params = {
            "latitude": float(lat),
            "longitude": float(lon),
            "start_date": start_dt.date().isoformat(),
            "end_date": end_dt.date().isoformat(),
            "hourly": ",".join(weather_vars),
            "timezone": timezone,
        }

        resp = requests.get(url, params=params, timeout=request_timeout)

        if str(resp) != '<Response [200]>':
            print(resp)
            return resp

        resp.raise_for_status()

        payload = resp.json()
        hourly = payload.get("hourly", {})

        if not hourly or "time" not in hourly:
            continue

        weather_df = pd.DataFrame(
            {datetime_col: to_kyiv_tz(pd.Series(hourly["time"]))}
        ).dropna(subset=[datetime_col])

        if weather_df.empty:
            continue

        weather_df[datetime_col] = floor_hour_safe(weather_df[datetime_col])
        weather_df[lat_col] = float(lat)
        weather_df[lon_col] = float(lon)

        expected_len = len(weather_df)
        for col in weather_vars:
            values = hourly.get(col)
            if values is None:
                weather_df[col] = np.nan
            else:
                padded = list(values[:expected_len])
                if len(padded) < expected_len:
                    padded.extend([np.nan] * (expected_len - len(padded)))
                weather_df[col] = padded

        weather_frames.append(weather_df)

    if not weather_frames:
        return out_df

    all_weather = (
        pd.concat(weather_frames, ignore_index=True)
        .drop_duplicates(subset=[datetime_col, lat_col, lon_col], keep="last")
    )

    merged = out_df.merge(
        all_weather,
        on=[datetime_col, lat_col, lon_col],
        how="left",
        suffixes=("", "__new"),
    )

    for col in weather_vars:
        new_col = f"{col}__new"
        if new_col in merged.columns:
            merged[col] = merged[col].where(~merged[col].isna(), merged[new_col])
            merged.drop(columns=[new_col], inplace=True)

    return merged

weather_vars_archive = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "surface_pressure",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "precipitation",
    "rain",
    "snowfall",
]

def read_txt() -> str:
    with open("weather_func_status.txt", "r", encoding="utf-8") as f:
        return f.read()

def smart_requests(df, eic_code_name = 'EIC-код',
    lat_col: str = "GPS-координати - Широта",
    lon_col: str = "GPS-координати - Довгота",
    sleep_time = 50,
    weather_vars = weather_vars_archive):
    try:
        df_with_weather = df[~df['temperature_2m'].isna()]
        df_no_weather = df[df['temperature_2m'].isna()]
        print(f"Gas stations with weather: {len(df_with_weather[eic_code_name].unique())}\nGas stations with no weather: {len(df_no_weather[eic_code_name].unique())}")
    except KeyError:
        df_with_weather = pd.DataFrame()
        df_no_weather = df.copy()
        print(f"Gas stations with weather: {0}\nGas stations with no weather: {len(df_no_weather[eic_code_name].unique())}")
    first_run = True
    for gas_station in tqdm(df_no_weather[eic_code_name].unique(), desc="Gas stations"):
        if first_run:
            first_run = False
        else:
            txt = read_txt()
            if txt == 'STOP':
                print(f"Stop request received, stopping now")
                break
            time.sleep(sleep_time)
        try:
            df_weathered = add_openmeteo_weather_sync(
                df=df_no_weather[df_no_weather[eic_code_name]==gas_station],
                weather_vars=weather_vars,
                lat_col=lat_col,
                lon_col=lon_col,
            )
        except Exception as e:
            print(e)
            print(f"\nStopping now")
            break
        if not isinstance(df_weathered, pd.DataFrame):
            try:
                print(f"{df_weathered.status_code}\n{df_weathered.text}")
            except Exception as e:
                print(e)
            print(f"\nStopping now")
            break
        else:
            df_with_weather = pd.concat([df_with_weather, df_weathered])
            df_no_weather = df_no_weather[df_no_weather[eic_code_name] != gas_station]
            df_save = pd.concat([df_with_weather, df_no_weather])
            # df_save.to_parquet(checkpoint_path)
    df = pd.concat([df_with_weather, df_no_weather])
    # df.to_parquet(path_out)
    return df