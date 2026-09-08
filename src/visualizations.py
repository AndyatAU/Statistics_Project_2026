"""Student C: charts for the global-temperature project."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


def plot_temperature_trend(annual_df: pd.DataFrame) -> plt.Figure:
    """Return an annual-temperature line chart with a 10-year rolling average."""
    df = annual_df.copy()
    df["rolling_average"] = df["annual_temperature_c"].rolling(10, center=True).mean()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df["year"], df["annual_temperature_c"], alpha=0.45, color="#4c78a8", label="Annual average")
    ax.plot(df["year"], df["rolling_average"], linewidth=2.5, color="#e45756", label="10-year rolling average")
    ax.set(title="Global Land and Ocean Average Temperature", xlabel="Year", ylabel="Temperature (deg C)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_temperature_distribution(annual_df: pd.DataFrame) -> plt.Figure:
    """Return a histogram of annual average temperatures."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(annual_df["annual_temperature_c"], bins=20, color="#72b7b2", edgecolor="white")
    ax.set(title="Distribution of Annual Average Temperature", xlabel="Temperature (deg C)", ylabel="Number of years")
    fig.tight_layout()
    return fig


def create_interactive_trend(annual_df: pd.DataFrame):
    """Return an interactive Plotly temperature-trend chart."""
    return px.line(
        annual_df,
        x="year",
        y="annual_temperature_c",
        title="Interactive Global Temperature Trend",
        labels={"year": "Year", "annual_temperature_c": "Temperature (deg C)"},
    )
