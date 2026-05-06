import pandas as pd
from data_procesing_pipeline import *

def extend_datetime_range(
    df: pd.DataFrame,
    start_dt,
    end_dt,
    eic_col: str = "EIC-код",
    dt_col: str = "Дата",
    value_col: str = "Sum of кВт",
    freq: str = "H",
) -> pd.DataFrame:
    """
    Extends the datetime range for each unique EIC code.

    Existing values are preserved.
    Future timestamps are added with NaN in the value column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    start_dt : str or datetime-like
        Start of the new datetime range.
        Must be >= current minimum timestamp.
    end_dt : str or datetime-like
        End of the new datetime range.
    eic_col : str
        Name of EIC code column.
    dt_col : str
        Name of datetime column.
    value_col : str
        Name of value column.
    freq : str
        Frequency of timestamps (default hourly).

    Returns
    -------
    pd.DataFrame
    """

    df = df.copy()

    # Ensure datetime
    df[dt_col] = pd.to_datetime(df[dt_col])

    start_dt = pd.to_datetime(start_dt)
    end_dt = pd.to_datetime(end_dt)

    current_min = df[dt_col].min()

    if start_dt < current_min:
        raise ValueError(
            f"start_dt ({start_dt}) cannot be earlier than "
            f"existing minimum timestamp ({current_min})"
        )

    # Create full datetime range
    full_dates = pd.date_range(start=start_dt, end=end_dt, freq=freq)

    # Unique EICs
    unique_eics = df[eic_col].unique()

    # Full cartesian product
    full_index = pd.MultiIndex.from_product(
        [unique_eics, full_dates],
        names=[eic_col, dt_col]
    )

    full_df = pd.DataFrame(index=full_index).reset_index()

    # Merge with original data
    result = full_df.merge(
        df,
        on=[eic_col, dt_col],
        how="left"
    )

    # Sort
    result = result.sort_values([eic_col, dt_col]).reset_index(drop=True)

    return result

def inference_preprocessing_full(df, start_dt, end_dt, ohe = False):
    extended_df = extend_datetime_range(
        df,
        start_dt=start_dt,
        end_dt=end_dt
    )
    df = pd.concat([df, extended_df]).copy()

    static_path = "../data/raw/8month2025.xlsx"
    static_data_dict = get_static_data_dict(path=static_path)

    df = df.merge(static_data_dict, on='EIC-код', how='left', suffixes=('', '_dict'))

    df = add_solar(df)

    df = add_max_power(df)

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

    df = add_weather_wrapper(df, no_weather_path, weather_path, checkpoint_path, weather_cols_to_create_all)

    df['datetime'] = pd.to_datetime(df['datetime'])

    df = add_ev_power(df, static_path, "../data/additional_data/EV_chargers.xlsx")

    df = add_calendar_features(df)

    df = build_time_index(df)

    df = df[[
                'eic_code', 'datetime', 'sum_of_kWh',
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

    # df = trimm_stations(df)

    df = add_prices(df)

    static_cols = [
        'latitude', 'longitude', 'eic_code', 'dso_desc', 'station_type', 'oblast'
    ]

    calendar_cols = [
        'Month', 'Day', 'Hour', 'day_of_week', 'season'
    ]

    GLOBAL_MIN_DT = df["datetime"].min()

    df = df.sort_values(['eic_code', 'datetime'])

    df['time_idx'] = (
            (df['datetime'] - GLOBAL_MIN_DT)
            .dt.total_seconds() // 3600
    ).astype(int)

    print(f"Number of missing timesteps: {len(find_missing_timesteps(df))}")

    if ohe:
        df = pd.get_dummies(
            df,
            columns=cols_to_ohe,
            drop_first=False
        )

    return df