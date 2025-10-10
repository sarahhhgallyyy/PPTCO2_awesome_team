"""Utilities for discovering and loading experiment datasets."""

from __future__ import annotations

from dataclasses import dataclass
from io import IOBase
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

# Default data directory lives at the repository root.
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "Data"


@dataclass(frozen=True)
class DatasetInfo:
    """Lightweight container describing a dataset file."""

    name: str
    path: Path
    size_bytes: int

    @property
    def display_label(self) -> str:
        """Return a friendly label for dropdowns."""
        size_mb = self.size_bytes / (1024 * 1024)
        return f"{self.name} ({size_mb:.1f} MB)"


def list_data_files(
    roots: Optional[Sequence[Path]] = None,
    patterns: Iterable[str] = ("*.csv", "*.CSV", "*.txt"),
) -> list[DatasetInfo]:
    """Return dataset descriptors for available files.

    Args:
        roots: Directories to search; defaults to the main data directory.
        patterns: Glob patterns for matching files.
    """
    search_roots: Sequence[Path] = roots or (DEFAULT_DATA_DIR,)
    infos: list[DatasetInfo] = []

    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue
        for pattern in patterns:
            for path in root.glob(pattern):
                infos.append(
                    DatasetInfo(
                        name=path.name,
                        path=path,
                        size_bytes=path.stat().st_size,
                    )
                )

    infos.sort(key=lambda info: info.name.lower())
    return infos


def load_dataset(
    source: Path | str | IOBase,
    *,
    decimal: str = ",",
    separator: str = ",",
    parse_dates: bool = True,
    datetime_column: int | str = 0,
    rename_datetime_column: str = "DateTime",
    drop_empty_columns: bool = True,
) -> pd.DataFrame:
    """Load a dataset into a DataFrame with a datetime index.

    Args:
        path: File path to load.
        decimal: Decimal separator used in numeric columns.
        separator: Column separator; default comma.
        parse_dates: Whether to parse the datetime column.
        datetime_column: Column to treat as datetime during parsing.
        rename_datetime_column: Friendly name for the index.
        drop_empty_columns: If True, drop columns that are completely NaN.
    """
    read_kwargs: dict = {
        "sep": separator,
        "decimal": decimal,
        "na_values": ["", " ", "-"],
    }

    if parse_dates:
        read_kwargs["parse_dates"] = [datetime_column]

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, **read_kwargs)
    elif isinstance(source, IOBase):
        df = pd.read_csv(source, **read_kwargs)
    else:
        # Allow objects similar to Streamlit's UploadedFile.
        if hasattr(source, "read"):
            df = pd.read_csv(source, **read_kwargs)
        else:
            raise TypeError(f"Unsupported source type: {type(source)!r}")

    if parse_dates:
        df = df.set_index(df.columns[datetime_column])
        df.index.name = rename_datetime_column
    else:
        df.index.name = df.columns[datetime_column]

    # Convert remaining columns to numeric when possible.
    value_columns = df.columns.difference([df.index.name]) if parse_dates else df.columns
    df[value_columns] = df[value_columns].apply(pd.to_numeric, errors="coerce")

    if drop_empty_columns:
        df = df.dropna(axis=1, how="all")

    df = df.sort_index()
    return df


def slice_time_window(
    df: pd.DataFrame,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Return a time-sliced view of the dataframe."""
    if start is None and end is None:
        return df
    return df.loc[start:end]
