import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Global Temperature Dashboard", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def find_file(candidates):
    search_folders = [
        PROJECT_ROOT,
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "data" / "datasets"
    ]
    for folder in search_folders:
        for name in candidates:
            path = folder / name
            if path.exists():
                return path
    return None

@st.cache_data
def load_country_data():
    path = find_file([
        "GlobalLandTemperaturesByCountry_clean.csv",
        "GlobalLandTemperaturesByCountry.csv"
    ])
    if path is None:
        return None
    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    df["Year"] = df["dt"].dt.year
    return df

@st.cache_data
def load_global_data():
    path = find_file(["GlobalTemperatures.csv"])
    if path is None:
        return None
    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    return df

country_df = load_country_data()
global_df = load_global_data()

st.title("🌍 Global Temperature Statistical Dashboard")
st.caption("Student C — Data Visualization")

if country_df is None:
    st.error(
        "Country CSV not found. Put GlobalLandTemperaturesByCountry.csv "
        "or GlobalLandTemperaturesByCountry_clean.csv in the same folder as app.py."
    )
    st.stop()

page = st.sidebar.radio(
    "Choose page",
    ["Overview", "Distributions", "Country Trends", "Hypothesis Testing"]
)

all_countries = sorted(country_df["Country"].dropna().unique().tolist())
default_countries = [c for c in ["Thailand", "Japan", "Australia"] if c in all_countries]

if page == "Overview":
    st.header("1. Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(country_df):,}")
    c2.metric("Countries", f"{country_df['Country'].nunique():,}")
    c3.metric(
        "Temperature records",
        f"{country_df['AverageTemperature'].notna().sum():,}"
    )

    st.subheader("Country summary")
    chosen = st.multiselect(
        "Countries",
        all_countries,
        default=default_countries
    )

    overview = country_df[country_df["Country"].isin(chosen)]
    summary = (
        overview.groupby("Country")["AverageTemperature"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .round(2)
    )
    st.dataframe(summary, use_container_width=True)

    st.subheader("Boxplot")
    if chosen:
        fig, ax = plt.subplots(figsize=(9, 5))
        values = [
            overview.loc[overview["Country"] == c, "AverageTemperature"].dropna()
            for c in chosen
        ]
        ax.boxplot(values, tick_labels=chosen)
        ax.set_ylabel("Average Temperature (°C)")
        ax.set_title("Average Temperature by Country")
        ax.grid(axis="y", alpha=0.25)
        st.pyplot(fig)

    st.subheader("Correlation heatmap")
    if global_df is None:
        st.info("Add GlobalTemperatures.csv to display the correlation heatmap.")
    else:
        numeric_cols = global_df.select_dtypes(include=np.number).columns.tolist()
        corr = global_df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(corr.values, aspect="auto")
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=90)
        ax.set_yticklabels(numeric_cols)

        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                value = corr.iloc[i, j]
                if pd.notna(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)

        fig.colorbar(im, ax=ax, label="Correlation")
        ax.set_title("Correlation Heatmap")
        fig.tight_layout()
        st.pyplot(fig)

elif page == "Distributions":
    st.header("2. Distributions")

    country = st.selectbox("Choose country", all_countries)
    data = country_df.loc[
        country_df["Country"] == country,
        "AverageTemperature"
    ].dropna().values

    if len(data) == 0:
        st.warning("No valid temperature data for this country.")
        st.stop()

    distribution = st.selectbox(
        "Fitted distribution",
        ["Normal", "Exponential", "Gamma", "Lognormal", "Uniform"]
    )

    dist_map = {
        "Normal": stats.norm,
        "Exponential": stats.expon,
        "Gamma": stats.gamma,
        "Lognormal": stats.lognorm,
        "Uniform": stats.uniform
    }

    dist = dist_map[distribution]
    params = dist.fit(data)
    x = np.linspace(data.min(), data.max(), 400)
    pdf = dist.pdf(x, *params)

    ks_stat, p_value = stats.kstest(data, dist.cdf, args=params)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Histogram + fitted distribution")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(data, bins=30, density=True, alpha=0.65, edgecolor="black")
        ax.plot(x, pdf, linewidth=2, label=distribution)
        ax.set_xlabel("Average Temperature (°C)")
        ax.set_ylabel("Density")
        ax.legend()
        st.pyplot(fig)

        st.write(f"**KS statistic:** {ks_stat:.4f}")
        st.write(f"**KS p-value:** {p_value:.6g}")

    with col2:
        st.subheader("Normal Q-Q plot")
        fig, ax = plt.subplots(figsize=(6, 5))
        stats.probplot(data, dist="norm", plot=ax)
        ax.set_title(f"{country} — Normal Q-Q Plot")
        st.pyplot(fig)

    st.info(
        "For the Q-Q plot: points close to the line suggest data are closer to a Normal distribution."
    )

elif page == "Country Trends":
    st.header("3. Country Trends")

    chosen = st.multiselect(
        "Choose countries",
        all_countries,
        default=default_countries
    )

    if chosen:
        trend = (
            country_df[country_df["Country"].isin(chosen)]
            .dropna(subset=["Year", "AverageTemperature"])
            .groupby(["Year", "Country"], as_index=False)["AverageTemperature"]
            .mean()
        )

        fig = px.line(
            trend,
            x="Year",
            y="AverageTemperature",
            color="Country",
            title="Average Temperature Trend",
            labels={"AverageTemperature": "Average Temperature (°C)"}
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "Hypothesis Testing":
    st.header("4. Hypothesis Testing")
    st.caption(
        "This page visualizes/repeats Student B's statistical tests for interactive use."
    )

    test_type = st.radio("Test", ["Welch's t-test", "One-way ANOVA"])

    if test_type == "Welch's t-test":
        col1, col2 = st.columns(2)
        c1 = col1.selectbox(
            "Country 1",
            all_countries,
            index=all_countries.index("Thailand") if "Thailand" in all_countries else 0
        )
        c2_default = all_countries.index("Japan") if "Japan" in all_countries else min(1, len(all_countries)-1)
        c2 = col2.selectbox("Country 2", all_countries, index=c2_default)

        x = country_df.loc[country_df["Country"] == c1, "AverageTemperature"].dropna()
        y = country_df.loc[country_df["Country"] == c2, "AverageTemperature"].dropna()

        t_stat, p_value = stats.ttest_ind(x, y, equal_var=False)

        st.metric("T-statistic", f"{t_stat:.4f}")
        st.metric("P-value", f"{p_value:.6g}")

        if p_value < 0.05:
            st.success(
                "p < 0.05 → Significant difference in mean temperature between the two groups."
            )
        else:
            st.info(
                "p ≥ 0.05 → No statistically significant difference detected."
            )

    else:
        chosen = st.multiselect(
            "Choose at least 3 countries",
            all_countries,
            default=default_countries
        )

        if len(chosen) < 3:
            st.warning("Please choose at least 3 countries.")
        else:
            groups = [
                country_df.loc[
                    country_df["Country"] == c,
                    "AverageTemperature"
                ].dropna()
                for c in chosen
            ]

            f_stat, p_value = stats.f_oneway(*groups)

            st.metric("F-statistic", f"{f_stat:.4f}")
            st.metric("P-value", f"{p_value:.6g}")

            if p_value < 0.05:
                st.success(
                    "p < 0.05 → At least one group mean differs significantly."
                )
            else:
                st.info(
                    "p ≥ 0.05 → No statistically significant difference detected among the selected groups."
                )
