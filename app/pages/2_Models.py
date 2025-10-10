from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    from app.models import MODEL_REGISTRY, get_model_module
    from app.models.heating_free import HeatingFreeParams
    from app.services import data_io
except ModuleNotFoundError:
    APP_ROOT = Path(__file__).resolve().parents[1]
    PROJECT_ROOT = APP_ROOT.parent
    for candidate in (PROJECT_ROOT, APP_ROOT):
        if str(candidate) not in sys.path:
            sys.path.append(str(candidate))
    from models import MODEL_REGISTRY, get_model_module  # type: ignore
    from models.heating_free import HeatingFreeParams  # type: ignore
    from services import data_io  # type: ignore


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


def _select_default_columns(columns) -> list[str]:
    candidates = [col for col in columns if col.upper().startswith("T")]
    return candidates[: min(len(candidates), 16)]


def _filter_dataframe(df: pd.DataFrame, selected_columns: list[str]) -> Tuple[pd.DataFrame, int]:
    subset = df.loc[:, selected_columns]
    mask = subset.notna().all(axis=1)
    filtered = subset[mask]
    dropped = int((~mask).sum())
    return filtered, dropped


def _compute_rmse(simulated: pd.DataFrame, measured: pd.DataFrame) -> Tuple[np.ndarray, float]:
    diff = simulated.to_numpy() - measured.to_numpy()
    per_column = np.sqrt(np.nanmean(diff**2, axis=0))
    overall = float(np.sqrt(np.nanmean(diff**2)))
    return per_column, overall


def _store_parameters(params: HeatingFreeParams) -> None:
    st.session_state.setdefault("heating_free_params", {})
    st.session_state["heating_free_params"] = {
        "alpha": float(params.alpha),
        "T_outside": float(params.T_outside),
        "epsilon": float(params.epsilon),
    }


def _get_parameter_state() -> Dict[str, float]:
    defaults = HeatingFreeParams()
    stored = st.session_state.get(
        "heating_free_params",
        {
            "alpha": defaults.alpha,
            "T_outside": defaults.T_outside,
            "epsilon": defaults.epsilon,
        },
    )
    # ensure floats
    return {
        "alpha": float(stored["alpha"]),
        "T_outside": float(stored["T_outside"]),
        "epsilon": float(stored["epsilon"]),
    }


def render_heating_free(df: pd.DataFrame, tank_columns: list[str], convert_to_kelvin: bool) -> None:
    model_module = get_model_module("heating_free")
    param_state = _get_parameter_state()

    with st.expander("Model Parameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        alpha = c1.number_input(
            "α (W/m²·K)",
            min_value=0.0,
            max_value=20000.0,
            step=50.0,
            value=param_state["alpha"],
            format="%.2f",
        )
        t_outside = c2.number_input(
            "Ambient temperature (K)",
            min_value=150.0,
            max_value=350.0,
            step=1.0,
            value=param_state["T_outside"],
            format="%.2f",
        )
        epsilon = c3.number_input(
            "Emissivity (ε)",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=param_state["epsilon"],
            format="%.3f",
        )

    selected_df, dropped_rows = _filter_dataframe(df, tank_columns)
    if selected_df.empty:
        st.error("No valid data after removing rows with missing values.")
        return

    if dropped_rows:
        st.info(f"Dropped {dropped_rows} rows containing missing values.")

    measured_display = selected_df
    measured_model = selected_df.copy()
    if convert_to_kelvin:
        measured_model = measured_model + 273.15

    params = HeatingFreeParams(alpha=alpha, T_outside=t_outside, epsilon=epsilon)

    simulation_key = "heating_free_simulation"
    result_container = st.empty()
    metrics_container = st.empty()

    if st.button("Run Simulation", type="primary"):
        with st.spinner("Running simulation..."):
            try:
                sim_df = model_module.simulate(measured_model, tank_columns, params)
                if convert_to_kelvin:
                    sim_display = sim_df - 273.15
                else:
                    sim_display = sim_df
                column_rmse, overall_rmse = _compute_rmse(sim_df, measured_model)
                st.session_state[simulation_key] = {
                    "measured_display": measured_display,
                    "sim_display": sim_display,
                    "rmse": column_rmse,
                    "overall_rmse": overall_rmse,
                    "columns": tank_columns,
                    "params": params,
                }
                _store_parameters(params)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Simulation failed: {exc}")

    if simulation_key in st.session_state:
        sim_state = st.session_state[simulation_key]
        comparison_df = {}
        for idx, col in enumerate(sim_state["columns"]):
            comparison_df[f"{col} (measured)"] = sim_state["measured_display"][col]
            comparison_df[f"{col} (sim)"] = sim_state["sim_display"][col]
        comparison_df = pd.DataFrame(comparison_df, index=sim_state["measured_display"].index)
        result_container.subheader("Simulation Output")
        result_container.line_chart(comparison_df)

        metrics_df = pd.DataFrame(
            {
                "Tank": sim_state["columns"],
                "RMSE (K)": sim_state["rmse"],
            }
        )
        metrics_container.subheader("Error Metrics")
        metrics_container.dataframe(metrics_df)
        metrics_container.metric("Overall RMSE (K)", f"{sim_state['overall_rmse']:.3f}")

    if st.button("Fit Parameters", type="secondary"):
        with st.spinner("Optimizing parameters..."):
            try:
                fitted_params, diagnostics = model_module.fit_parameters(measured_model, tank_columns, initial_guess=params)
                _store_parameters(fitted_params)
                st.success(
                    f"Fit complete: α={fitted_params.alpha:.1f}, "
                    f"T_outside={fitted_params.T_outside:.2f}, ε={fitted_params.epsilon:.3f}"
                )
                st.caption(f"Solver status: {diagnostics.get('message', 'n/a')}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Parameter fitting failed: {exc}")


def render():
    st.title("Models")
    st.caption("Configure and run simulations using the available models.")

    models = list(MODEL_REGISTRY.values())
    if not models:
        st.warning("No models registered.")
        return

    model_labels = {info.label: info.key for info in models}
    selected_label = st.selectbox("Model", options=list(model_labels.keys()))
    selected_key = model_labels[selected_label]
    model_info = MODEL_REGISTRY[selected_key]
    st.write(model_info.description)

    available_datasets = data_io.list_data_files()

    with st.sidebar:
        st.header("Data Options")
        decimal = st.selectbox("Decimal separator", [",", "."], index=0)
        separator = st.text_input("Field separator", value=",", max_chars=3)
        parse_dates = st.checkbox("Parse first column as datetime", value=True)
        datetime_column = st.number_input("Datetime column index", min_value=0, value=0, step=1)

    col_select, col_upload = st.columns((2, 1))
    selected_dataset = None

    with col_select:
        if available_datasets:
            selected_dataset = st.selectbox(
                "Dataset",
                options=available_datasets,
                format_func=lambda info: info.display_label,
            )
        else:
            st.warning("No local datasets detected in the Data directory.")

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=["csv", "txt"],
            help="Uploaded files remain in memory for this session only.",
        )

    df: pd.DataFrame | None = None
    source_name: str | None = None

    if uploaded_file is not None:
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
    elif selected_dataset is not None:
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
        st.info("Load a dataset to configure the model.")
        return

    st.subheader("Dataset")
    st.write(f"**Source:** {source_name}")
    st.caption(f"Rows: {len(df):,} · Columns: {df.shape[1]:,}")

    if isinstance(df.index, pd.DatetimeIndex):
        st.caption(
            f"Time span: {df.index.min().to_pydatetime()} → {df.index.max().to_pydatetime()}"
        )

    candidate_columns = _select_default_columns(df.columns)
    tank_columns = st.multiselect(
        "Tank temperature columns",
        options=df.columns.tolist(),
        default=candidate_columns,
        help="Select the columns representing tank wall temperatures.",
    )

    if not tank_columns:
        st.warning("Select at least one tank temperature column.")
        return

    convert_choice = st.radio(
        "Temperature data stored as",
        options=["Celsius", "Kelvin"],
        index=0,
        help="The model expects Kelvin internally; values in Celsius will be offset by +273.15.",
    )
    convert_to_kelvin = convert_choice == "Celsius"

    if selected_key == "heating_free":
        render_heating_free(df, tank_columns, convert_to_kelvin)
    else:
        st.info("Selected model is not yet integrated into the UI.")


render()
