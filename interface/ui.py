import sys
from pathlib import Path

# Ensure the directory containing this file is on sys.path
APP_DIR = Path(__file__).parent.resolve()
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st
import pandas as pd
import importlib
import io
import calendar
from datetime import datetime, timedelta

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Electricity Forecasting",
    page_icon="⚡",
    layout="centered",
)

# ── Styling ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; }
        .stAlert { border-radius: 8px; }
        h1 { letter-spacing: -0.5px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Model registry ──────────────────────────────────────────────────────────────
# Add or remove models here.
# key   → displayed in the dropdown
# value → (module_name, function_name)
MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "XGBoost":  ("xgboost_inference", "predict_xgboost"),
    "Chronos ZS":     ("chronos_zs_inference",     "predict_chronos_zs"),
    "Chronos FT":     ("chronos_ft_inference",     "predict_chronos_ft"),
    "TCN":     ("tcn_inference",     "predict_tcn"),
    "Crossformer":     ("crossformer_inference",     "predict_crossformer"),
    # "Prophet": ("prophet_inference",  "predict_prophet"),
}


def load_model_fn(module_name: str, fn_name: str):
    """Dynamically import the inference module and return the predict function."""
    import traceback
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        # Check whether it's THIS module that's missing, or a dependency inside it
        if e.name == module_name:
            return None, (
                f"Module '{module_name}.py' not found on sys.path.\n\n"
                f"sys.path = {sys.path}"
            )
        else:
            return None, (
                f"'{module_name}.py' was found, but it failed to import "
                f"because a dependency is missing: **{e.name}**\n\n"
                f"Full traceback:\n```\n{traceback.format_exc()}\n```"
            )
    except Exception as e:
        return None, (
            f"'{module_name}.py' was found, but raised an error while importing:\n\n"
            f"```\n{traceback.format_exc()}\n```"
        )
    if not hasattr(module, fn_name):
        return None, f"Function '{fn_name}' not found inside '{module_name}.py'."
    return getattr(module, fn_name), None


def _find_last_timestamp(df: pd.DataFrame) -> datetime | None:
    """Return the latest datetime found in any datetime-like column, tz-stripped."""
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if not dt_cols:
        for c in df.columns:
            if df[c].dtype == object:
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notna().sum() > len(df) * 0.5:
                    dt_cols.append(c)
                    df = df.copy()
                    df[c] = parsed
    if not dt_cols:
        return None
    last_ts = max(df[c].max() for c in dt_cols)
    if pd.isna(last_ts):
        return None
    ts = pd.Timestamp(last_ts)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.to_pydatetime()


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Serialise a DataFrame to an in-memory Excel file and return raw bytes."""
    df = df.copy()
    # Excel does not support timezone-aware datetimes — strip tz info
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]) and df[col].dt.tz is not None:
            df[col] = df[col].dt.tz_localize(None)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Forecast")
    return buf.getvalue()


# ── UI ──────────────────────────────────────────────────────────────────────────

# Initialise forecast-window session state on first run
_now = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
_default_end = _now + timedelta(hours=24)
for _k, _v in [
    ("forecast_start_date", _now.date()),
    ("forecast_start_time", _now.time()),
    ("forecast_end_date",   _default_end.date()),
    ("forecast_end_time",   _default_end.time()),
    ("last_uploaded_filename", None),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

st.title("⚡ Electricity Forecasting")
st.caption("Upload historical data, choose a model, set the forecast window, and download results.")

st.divider()

# ── 1. File upload ──────────────────────────────────────────────────────────────
st.subheader("1 · Upload historical data")
uploaded_file = st.file_uploader(
    "Excel file (.xlsx / .xls)",
    type=["xlsx", "xls"],
    help="The file should contain at least the columns expected by the chosen model.",
)

df_input: pd.DataFrame | None = None

if uploaded_file is not None:
    try:
        df_input = pd.read_excel(uploaded_file)
        st.success(f"Loaded **{len(df_input):,}** rows × **{len(df_input.columns)}** columns.")
        with st.expander("Preview (first 10 rows)"):
            st.dataframe(df_input.head(10), use_container_width=True)

        # Auto-set forecast window to next full month on new file upload
        if st.session_state["last_uploaded_filename"] != uploaded_file.name:
            st.session_state["last_uploaded_filename"] = uploaded_file.name
            last_ts = _find_last_timestamp(df_input)
            if last_ts is not None:
                nm_year  = last_ts.year + (1 if last_ts.month == 12 else 0)
                nm_month = 1 if last_ts.month == 12 else last_ts.month + 1
                nm_start = datetime(nm_year, nm_month, 1, 0, 0, 0)
                nm_end   = datetime(nm_year, nm_month, calendar.monthrange(nm_year, nm_month)[1], 23, 0, 0)
                st.session_state["forecast_start_date"] = nm_start.date()
                st.session_state["forecast_start_time"] = nm_start.time()
                st.session_state["forecast_end_date"]   = nm_end.date()
                st.session_state["forecast_end_time"]   = nm_end.time()
    except Exception as exc:
        st.error(f"Could not read the file: {exc}")

st.divider()

# ── 2. Forecast window ──────────────────────────────────────────────────────────
st.subheader("2 · Forecast window")

col_start, col_end = st.columns(2)

with col_start:
    start_date = st.date_input("Start date", key="forecast_start_date")
    start_time = st.time_input("Start time", key="forecast_start_time", step=3600)

with col_end:
    end_date = st.date_input("End date", key="forecast_end_date")
    end_time = st.time_input("End time", key="forecast_end_time", step=3600)

start_dt = datetime.combine(start_date, start_time)
end_dt   = datetime.combine(end_date,   end_time)

if end_dt <= start_dt:
    st.warning("⚠️ End datetime must be after start datetime.")

st.divider()

# ── 3. Model selection ──────────────────────────────────────────────────────────
st.subheader("3 · Model")

model_label = st.selectbox(
    "Forecasting model",
    options=list(MODEL_REGISTRY.keys()),
    help="Each model has different data requirements. Check the model documentation if you get an error.",
)

st.divider()

# ── 4. Run forecast ─────────────────────────────────────────────────────────────
st.subheader("4 · Run")

run_btn = st.button(
    "▶ Generate forecast",
    type="primary",
    disabled=(df_input is None or end_dt <= start_dt),
    use_container_width=True,
)

if run_btn:
    module_name, fn_name = MODEL_REGISTRY[model_label]

    with st.spinner(f"Loading **{model_label}** model…"):
        predict_fn, load_err = load_model_fn(module_name, fn_name)

    if load_err:
        st.error(f"🔴 {load_err}")
    else:
        # Inference always starts from the next full hour after the last
        # timestamp in the uploaded file, regardless of the UI start picker.
        last_ts = _find_last_timestamp(df_input)
        if last_ts is not None:
            infer_start_dt = (last_ts + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        else:
            infer_start_dt = start_dt  # fallback if no datetime column found

        with st.spinner("Running inference…"):
            result = predict_fn(
                df_input.copy(),
                start_dt=infer_start_dt,
                end_dt=end_dt,
            )

        # ── Handle result ──────────────────────────────────────────────────────
        if isinstance(result, str):
            # Model returned an error message
            st.error(f"🔴 Model error:\n\n{result}")

        elif isinstance(result, pd.DataFrame):
            if result.empty:
                st.warning("The model returned an empty DataFrame. Nothing to download.")
            else:
                # Filter to the requested window [infer_start_dt, end_dt]
                if "datetime" in result.columns:
                    dt_col = pd.to_datetime(result["datetime"])
                    if isinstance(dt_col.dtype, pd.DatetimeTZDtype):
                        dt_col = dt_col.dt.tz_localize(None)
                    window_mask = (dt_col >= pd.Timestamp(infer_start_dt)) & \
                                  (dt_col <= pd.Timestamp(end_dt))
                    result = result[window_mask].reset_index(drop=True)

                if result.empty:
                    st.warning("No forecast rows fall within the selected window.")
                else:
                    st.success(f"✅ Forecast complete — **{len(result):,}** rows generated.")

                    with st.expander("Preview forecast (first 20 rows)"):
                        st.dataframe(result.head(20), use_container_width=True)

                    excel_bytes = df_to_excel_bytes(result)
                    filename = (
                        f"forecast_{model_label.lower().replace(' ', '_')}"
                        f"_{infer_start_dt.strftime('%Y%m%d_%H%M')}"
                        f"_{end_dt.strftime('%Y%m%d_%H%M')}.xlsx"
                    )
                    st.download_button(
                        label="⬇️ Download forecast (.xlsx)",
                        data=excel_bytes,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
        else:
            st.error(
                f"Unexpected return type from model: `{type(result).__name__}`. "
                "The predict function must return either a pandas DataFrame or an error string."
            )

# ── Footer ──────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Run with: `streamlit run app.py`")