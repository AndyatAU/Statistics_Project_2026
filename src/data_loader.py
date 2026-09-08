"""Student A: load and clean the global-temperature dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


TEMPERATURE_COLUMN = "LandAndOceanAverageTemperature"


def load_global_temperatures(file_path: str | Path = "GlobalTemperatures.csv") -> pd.DataFrame:
    """Load monthly global land-and-ocean temperatures and remove unusable rows."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

    df = pd.read_csv(path, parse_dates=["dt"])
    if TEMPERATURE_COLUMN not in df.columns:
        raise ValueError(f"{TEMPERATURE_COLUMN} is missing from {path.name}")

    cleaned = df.dropna(subset=[TEMPERATURE_COLUMN]).copy()
    cleaned = cleaned.rename(columns={"dt": "date", TEMPERATURE_COLUMN: "temperature_c"})
    cleaned["year"] = cleaned["date"].dt.year
    cleaned["month"] = cleaned["date"].dt.month
    return cleaned.sort_values("date").reset_index(drop=True)


def create_annual_temperatures(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Convert cleaned monthly observations to one mean temperature per year."""
    if not {"year", "temperature_c"}.issubset(monthly_df.columns):
        raise ValueError("Data must contain year and temperature_c columns.")
    return (
        monthly_df.groupby("year", as_index=False)["temperature_c"]
        .mean()
        .rename(columns={"temperature_c": "annual_temperature_c"})
    )
