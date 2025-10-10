from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

try:
    from app.services import data_io
    from app.services.data_io import DatasetInfo
except ModuleNotFoundError:  # Running via Streamlit with app/ on sys.path only
    APP_ROOT = Path(__file__).resolve().parents[1]
    if str(APP_ROOT) not in sys.path:
        sys.path.append(str(APP_ROOT))
    from services import data_io  # type: ignore
    from services.data_io import DatasetInfo  # type: ignore


@st.cache_data(show_spinner=False)
def _load_local_dataset(
    path_str: str,
    decimal: str,
    separator: str,
    parse_dates: bool,
    datetime_column: int,
) -> pd.DataFrame:
    return data_io.load_dataset(
        Path(path_str),
        decimal=decimal,
        separator=separator,
        parse_dates=parse_dates,
        datetime_column=datetime_column,
    )


@st.cache_data(show_spinner=False)
def _load_uploaded_dataset(
    file_bytes: bytes,
    decimal: str,
    separator: str,
    parse_dates: bool,
    datetime_column: int,
) -> pd.DataFrame:
    buffer = BytesIO(file_bytes)
    return data_io.load_dataset(
        buffer,
        decimal=decimal,
        separator=separator,
        parse_dates=parse_dates,
        datetime_column=datetime_column,
    )


def _derive_time_window(df: pd.DataFrame, start, end) -> pd.DataFrame:
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    return data_io.slice_time_window(df, start=start, end=end)


def _format_dataset_option(info: DatasetInfo) -> str:
    return info.display_label


def render():
    st.title("Data")
    st.caption("Select, inspect, and prepare experiment datasets.")

    available_datasets = data_io.list_data_files()

    with st.sidebar:
        st.header("Load Options")
        decimal = st.selectbox("Decimal separator", options=[",", "."], index=0)
        separator = st.text_input("Field separator", value=",", max_chars=3)
        parse_dates = st.checkbox("Parse first column as datetime", value=True)
        datetime_column = st.number_input("Datetime column index", min_value=0, value=0, step=1)

    col_select, col_upload = st.columns((2, 1))

    with col_select:
        selected_dataset = st.selectbox(
            "Available datasets",
            options=available_datasets,
            format_func=_format_dataset_option,
            index=0 if available_datasets else None,
        )

    with col_upload:
        uploaded_file = st.file_uploader(
            "Or upload CSV",
            type=["csv", "txt"],
            help="Uploaded data is kept in memory for the current session.",
        )

    df: pd.DataFrame | None = None
    source_name: str | None = None

    if uploaded_file:
        try:
            df = _load_uploaded_dataset(
                uploaded_file.getvalue(),
                decimal=decimal,
                separator=separator,
                parse_dates=parse_dates,
                datetime_column=int(datetime_column),
            )
            source_name = uploaded_file.name
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to load uploaded file: {exc}")
    elif selected_dataset:
        try:
            df = _load_local_dataset(
                str(selected_dataset.path),
                decimal=decimal,
                separator=separator,
                parse_dates=parse_dates,
                datetime_column=int(datetime_column),
            )
            source_name = selected_dataset.name
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to load dataset: {exc}")

    if df is None:
        st.info("Select a dataset or upload a file to explore the data.")
        return

    st.subheader("Dataset Overview")
    st.write(f"**Source:** {source_name}")

    if isinstance(df.index, pd.DatetimeIndex):
        t_min = df.index.min().to_pydatetime()
        t_max = df.index.max().to_pydatetime()
        st.caption(f"Time span: {t_min} → {t_max}")

        default_start = t_min
        default_end = t_max
        if t_min == t_max:
            start_dt, end_dt = default_start, default_end
            st.caption("Only one timestamp available; using full extent.")
        else:
            start_dt, end_dt = st.slider(
                "Time window",
                min_value=t_min,
                max_value=t_max,
                value=(default_start, default_end),
                format="YYYY-MM-DD HH:mm:ss",
            )
        df_window = _derive_time_window(df, start=start_dt, end=end_dt)
    else:
        df_window = df
        st.caption("No datetime index detected; skipping time window controls.")

    metric_cols = st.columns(3)
    metric_cols[0].metric("Rows", f"{len(df_window):,}")
    metric_cols[1].metric("Columns", f"{df_window.shape[1]:,}")
    metric_cols[2].metric("Missing values", f"{df_window.isna().sum().sum():,}")

    column_options = df_window.columns.tolist()
    default_columns = column_options[: min(10, len(column_options))]
    selected_columns = st.multiselect(
        "Columns to display",
        options=column_options,
        default=default_columns,
    )

    if not selected_columns:
        st.warning("Select at least one column to display.")
        return

    st.subheader("Data Preview")
    st.dataframe(df_window[selected_columns].head(200))


render()
