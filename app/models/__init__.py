"""Model registry and loader utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from . import heating_free


@dataclass(frozen=True)
class ModelInfo:
    """Metadata describing a simulation model."""

    key: str
    label: str
    description: str
    module: Any


MODEL_REGISTRY: Dict[str, ModelInfo] = {
    "heating_free": ModelInfo(
        key="heating_free",
        label="Heating / Free Cooling",
        description=(
            "Single-phase model capturing conductive and radiative losses "
            "based on the legacy Heating_up implementation."
        ),
        module=heating_free,
    ),
}


def get_model_module(key: str):
    """Return the requested model module."""
    info = MODEL_REGISTRY.get(key)
    if info is None:
        raise KeyError(f"Unknown model key: {key}")
    return info.module


__all__ = ["ModelInfo", "MODEL_REGISTRY", "get_model_module"]
