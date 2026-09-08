"""Statistical analysis for Kaggle's Climate Change Global Temperature Data.

Run from this folder:
    python climate_analysis.py

Optional:
    python climate_analysis.py --input GlobalTemperatures.csv --output results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_FILE = "GlobalTemperatures.csv"
TEMPERATURE_COLUMN = "LandAndOceanAverageTemperature"


def descriptive_stats(values: pd.Series) -> dict[str, float]:
    """Return the descriptive statistics required for the project."""
    values = values.dropna()
    modes = values.mode()
    return {
        "count": float(values.size),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "mode": float(modes.iloc[0]) if not modes.empty else np.nan,
        "standard_deviation": float(values.std(ddof=1)),
        "variance": float(values.var(ddof=1)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "range": float(values.max() - values.min()),
        "iqr": float(values.quantile(0.75) - values.quantile(0.25)),
        "skewness": float(stats.skew(values, bias=False)),
        "kurtosis": float(stats.kurtosis(values, bias=False)),
    }


def bootstrap_mean_ci(
    values: pd.Series, n_bootstrap: int = 10_000, confidence: float = 0.95
) -> tuple[float, float]:
    """Calculate a non-parametric bootstrap confidence interval for the mean."""
    data = values.dropna().to_numpy()
    rng = np.random.default_rng(42)  # Reproducible results.
    samples = rng.choice(data, size=(n_bootstrap, len(data)), replace=True)
    means = samples.mean(axis=1)
    alpha = (1 - confidence) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse global temperature data.")
    parser.add_argument("--input", default=DEFAULT_FILE, help="Path to GlobalTemperatures.csv")
    parser.add_argument("--output", default="results", help="Folder for charts and results")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Cannot find {input_path}. Put GlobalTemperatures.csv beside this script "
            "or pass --input PATH_TO_FILE."
        )

    df = pd.read_csv(input_path, parse_dates=["dt"])
    if TEMPERATURE_COLUMN not in df.columns:
        raise ValueError(f"Expected column {TEMPERATURE_COLUMN!r}, found: {list(df.columns)}")

    # The land-and-ocean series begins in 1850. Remove unavailable monthly values.
    df = df.dropna(subset=[TEMPERATURE_COLUMN]).copy()
    df["Year"] = df["dt"].dt.year
    annual = df.groupby("Year", as_index=False)[TEMPERATURE_COLUMN].mean()
    temperatures = annual[TEMPERATURE_COLUMN]

    summary = descriptive_stats(temperatures)
    ci_low, ci_high = bootstrap_mean_ci(temperatures)

    # Linear-regression slope tests whether the long-run annual trend differs from zero.
    trend = stats.linregress(annual["Year"], temperatures)
    z_scores = stats.zscore(temperatures, nan_policy="omit")
    outliers = annual.loc[np.abs(z_scores) > 3, ["Year", TEMPERATURE_COLUMN]]
    normality_stat, normality_p = stats.normaltest(temperatures)

    results = [
        "GLOBAL LAND AND OCEAN TEMPERATURE ANALYSIS",
        "=" * 45,
        "",
        "Descriptive statistics for annual average temperature (deg C):",
        *[f"{key.replace('_', ' ').title()}: {value:.4f}" for key, value in summary.items()],
        "",
        f"95% bootstrap CI for the mean: ({ci_low:.4f}, {ci_high:.4f}) deg C",
        f"Normality test: statistic={normality_stat:.4f}, p-value={normality_p:.6g}",
        f"Linear trend: {trend.slope:.4f} deg C per year ({trend.slope * 100:.3f} deg C per century)",
        f"Trend test p-value: {trend.pvalue:.6g}",
        "",
        "Interpretation:",
        (
            "The temperature trend is statistically significant (p < 0.05)."
            if trend.pvalue < 0.05
            else "The temperature trend is not statistically significant at alpha = 0.05."
        ),
        (
            "Annual temperatures do not pass the normality test (p < 0.05)."
            if normality_p < 0.05
            else "Annual temperatures do not show evidence against normality (p >= 0.05)."
        ),
        "",
        "Potential outliers (absolute z-score > 3):",
        outliers.to_string(index=False) if not outliers.empty else "None",
    ]
    (output_path / "statistical_results.txt").write_text("\n".join(results), encoding="utf-8")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(annual["Year"], temperatures, color="#2166ac", linewidth=1.4, label="Annual average")
    ax.plot(
        annual["Year"],
        trend.intercept + trend.slope * annual["Year"],
        color="#b2182b",
        linewidth=2,
        label=f"Linear trend ({trend.slope * 100:.2f} deg C/century)",
    )
    ax.set(title="Global Land and Ocean Average Temperature", xlabel="Year", ylabel="Temperature (deg C)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path / "temperature_trend.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(temperatures, bins=20, density=True, color="#67a9cf", edgecolor="white", label="Annual temperatures")
    x = np.linspace(temperatures.min(), temperatures.max(), 300)
    ax.plot(x, stats.norm.pdf(x, temperatures.mean(), temperatures.std(ddof=1)), color="#b2182b", label="Normal fit")
    ax.set(title="Distribution of Annual Average Temperature", xlabel="Temperature (deg C)", ylabel="Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path / "temperature_distribution.png", dpi=200)
    plt.close(fig)

    print("Analysis complete.")
    print(f"Results: {output_path / 'statistical_results.txt'}")
    print(f"Charts:   {output_path / 'temperature_trend.png'} and temperature_distribution.png")


if __name__ == "__main__":
    main()
