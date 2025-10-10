"""Free cooling / heating model based on the legacy Heating_up script."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import least_squares

# Physical constants (kept consistent with the legacy script)
HEAT_CAPACITY_GLASS = 0.840  # J/(g*K)
DENSITY_GLASS = 2.2  # g/mL
TANK_VOID = 10.0  # mL (void part)
VOID_FRACTION = 0.10  # dimensionless
TANK_VOLUME = TANK_VOID / VOID_FRACTION  # mL
TANK_SURFACE = 0.0065  # m^2
BOLTZMANN_CONSTANT = 5.67e-8  # W/m^2/K^4


@dataclass
class HeatingFreeParams:
    """Model parameters."""

    alpha: float = 2000.0  # W/m^2/K
    T_outside: float = 263.0  # K
    epsilon: float = 0.1  # dimensionless

    def as_array(self) -> np.ndarray:
        return np.array([self.alpha, self.T_outside, self.epsilon], dtype=float)

    @classmethod
    def from_iterable(cls, values: Sequence[float]) -> "HeatingFreeParams":
        alpha, t_out, eps = values
        return cls(alpha=float(alpha), T_outside=float(t_out), epsilon=float(eps))


PARAM_BOUNDS = np.array([[0.0, 20000.0], [200.0, 320.0], [0.0, 1.0]])


def _heatbalance(
    temperature_profile: np.ndarray,
    _t: float,
    params: np.ndarray,
) -> np.ndarray:
    alpha, T_outside, epsilon = params
    dTdt = np.zeros_like(temperature_profile)

    radiative_term = BOLTZMANN_CONSTANT * epsilon
    convective_term = TANK_SURFACE * alpha
    denominator = TANK_VOLUME * DENSITY_GLASS * HEAT_CAPACITY_GLASS

    # Apply the same equation to all tanks (no trunk/series modeling).
    dTdt[:] = (
        convective_term * (T_outside - temperature_profile)
        + radiative_term * (T_outside**4 - temperature_profile**4)
    ) / denominator

    return dTdt


def _prepare_time_axis(df: pd.DataFrame) -> np.ndarray:
    """Generate an evenly spaced time axis based on the dataframe index."""
    if isinstance(df.index, pd.DatetimeIndex):
        t_seconds = (df.index - df.index[0]).total_seconds().to_numpy()
        # Guard against non-monotonic timestamps by enforcing increasing order.
        if not np.all(np.diff(t_seconds) >= 0):
            raise ValueError("Datetime index must be monotonically increasing.")
        return t_seconds

    # Fallback: use sample number as a proxy for elapsed time.
    return np.arange(len(df), dtype=float)


def simulate(
    df: pd.DataFrame,
    tank_columns: Sequence[str],
    params: HeatingFreeParams,
    *,
    initial_temperature: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Run the ODE model for the provided dataset."""
    if not tank_columns:
        raise ValueError("At least one tank column is required for simulation.")

    missing = [col for col in tank_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in dataframe: {missing}")

    t_eval = _prepare_time_axis(df)
    measured = df.loc[:, tank_columns].to_numpy(dtype=float)

    if initial_temperature is None:
        initial_temperature = measured[0]
    else:
        initial_temperature = np.asarray(initial_temperature, dtype=float)
        if initial_temperature.shape != (len(tank_columns),):
            raise ValueError(
                "Initial temperature must have shape "
                f"({len(tank_columns)},), got {initial_temperature.shape}"
            )

    solution = odeint(
        _heatbalance,
        y0=initial_temperature,
        t=t_eval,
        args=(params.as_array(),),
    )

    sim_df = pd.DataFrame(solution, index=df.index, columns=tank_columns)
    return sim_df


def _residuals(
    param_array: np.ndarray,
    initial_temperature: np.ndarray,
    observed: np.ndarray,
    t_eval: np.ndarray,
) -> np.ndarray:
    params = HeatingFreeParams.from_iterable(param_array)
    sim = odeint(
        _heatbalance,
        y0=initial_temperature,
        t=t_eval,
        args=(params.as_array(),),
    )
    return (sim - observed).ravel()


def fit_parameters(
    df: pd.DataFrame,
    tank_columns: Sequence[str],
    *,
    initial_guess: HeatingFreeParams = HeatingFreeParams(),
    bounds: Optional[np.ndarray] = None,
) -> tuple[HeatingFreeParams, dict]:
    """Fit model parameters to the measured data."""
    if bounds is None:
        bounds = PARAM_BOUNDS

    available_bounds = np.array(bounds, dtype=float)
    if available_bounds.shape != (3, 2):
        raise ValueError("Bounds must have shape (3, 2).")

    t_eval = _prepare_time_axis(df)
    measured = np.asarray(df.loc[:, tank_columns].to_numpy(dtype=float))

    if measured.size == 0:
        raise ValueError("No measurements available for fitting.")

    y0 = np.asarray(measured[0], dtype=float)

    result = least_squares(
        _residuals,
        x0=initial_guess.as_array(),
        bounds=(available_bounds[:, 0], available_bounds[:, 1]),
        args=(y0, measured, t_eval),
    )

    fitted = HeatingFreeParams.from_iterable(result.x)
    diagnostics = {
        "cost": result.cost,
        "success": result.success,
        "message": result.message,
        "nfev": result.nfev,
        "njev": result.njev,
    }
    return fitted, diagnostics
