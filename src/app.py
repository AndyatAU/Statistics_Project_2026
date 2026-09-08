"""Student C: Streamlit dashboard for the climate-temperature dataset."""

from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import create_annual_temperatures, load_global_temperatures
from src.visualizations import (
    create_interactive_trend,
    plot_temperature_distribution,
    plot_temperature_trend,
)


st.set_page_config(page_title="Climate Data Detective", page_icon="🌡️", layout="wide")
st.title("Climate Data Detective")
st.caption("Global land and ocean temperature analysis, 1850 to 2015")

@st.cache_data
def get_data():
    monthly = load_global_temperatures(PROJECT_ROOT / "GlobalTemperatures.csv")
    return monthly, create_annual_temperatures(monthly)


monthly_data, annual_data = get_data()
view = st.sidebar.radio("Analysis", ["Overview", "Trend", "Distribution"])

if view == "Overview":
    st.header("Dataset overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Years analysed", annual_data["year"].nunique())
    col2.metric("Mean temperature", f"{annual_data['annual_temperature_c'].mean():.2f} deg C")
    col3.metric("Warmest year", int(annual_data.loc[annual_data['annual_temperature_c'].idxmax(), 'year']))
    st.dataframe(monthly_data[["date", "temperature_c"]].head(20), use_container_width=True)

elif view == "Trend":
    st.header("Temperature trend")
    st.pyplot(plot_temperature_trend(annual_data))
    st.plotly_chart(create_interactive_trend(annual_data), use_container_width=True)

else:
    st.header("Temperature distribution")
    st.pyplot(plot_temperature_distribution(annual_data))
    st.write(annual_data["annual_temperature_c"].describe())
