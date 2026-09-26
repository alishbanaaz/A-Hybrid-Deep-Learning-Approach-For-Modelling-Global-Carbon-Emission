"""
CO2 Hybrid Forecast — Streamlit app.

Same hybrid SARIMAX + LSTM pipeline as the FYP notebooks (see model_pipeline.py),
wrapped in a Streamlit UI so it can be deployed for free on Streamlit Community
Cloud (no credit card required).

Run locally with:
    streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import model_pipeline as mp

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "global-data-on-sustainable-energy.csv"

st.set_page_config(page_title="CO2 Hybrid Forecast", page_icon="🌍", layout="wide")

# --------------------------------------------------------------------------
# Session state — holds the working dataset and the last trained result so
# Streamlit's rerun-on-every-interaction model doesn't lose them.
# --------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = pd.read_csv(DEFAULT_DATA_PATH)
    st.session_state.source = "Original FYP dataset"
if "result" not in st.session_state:
    st.session_state.result = None


def load_default():
    st.session_state.df = pd.read_csv(DEFAULT_DATA_PATH)
    st.session_state.source = "Original FYP dataset"
    st.session_state.result = None


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🌍 CO₂ Hybrid Forecast — SARIMAX + LSTM")
st.caption(
    "Final Year Project by Hafiza Alishba Naaz — NUST Islamabad. "
    "Upload your own energy data and forecast future CO₂ emissions."
)

# --------------------------------------------------------------------------
# 1. Dataset
# --------------------------------------------------------------------------
st.header("1. Dataset")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded = st.file_uploader(
        "Upload a CSV to use your own data",
        type=["csv"],
        help=(
            "Required columns: Entity, Year, Value_co2_emissions_kt_by_country, "
            "gdp_per_capita, gdp_growth, "
            "Primary energy consumption per capita (kWh/person), "
            "Renewable energy share in the total final energy consumption (%)"
        ),
    )
    if uploaded is not None:
        try:
            new_df = pd.read_csv(uploaded)
            missing = mp.validate_schema(new_df)
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            else:
                st.session_state.df = new_df
                st.session_state.source = f"Uploaded: {uploaded.name}"
                st.session_state.result = None
                st.success("Now using your uploaded dataset.")
        except Exception as e:
            st.error(f"Could not read that CSV: {e}")

with col2:
    st.write("")
    st.write("")
    if st.button("Reset to Original Dataset"):
        load_default()
        st.rerun()

df = st.session_state.df
st.markdown(
    f"**Source:** {st.session_state.source} &nbsp;|&nbsp; "
    f"**Rows:** {len(df)} &nbsp;|&nbsp; "
    f"**Countries:** {df['Entity'].nunique()} &nbsp;|&nbsp; "
    f"**Years:** {int(df['Year'].min())}–{int(df['Year'].max())}"
)

# --------------------------------------------------------------------------
# 2. Train & Forecast
# --------------------------------------------------------------------------
st.header("2. Train & Forecast")

countries = mp.top_countries(df, n=15)
c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
with c1:
    country = st.selectbox("Country", countries)
with c2:
    test_years = st.number_input("Held-out test years", min_value=1, max_value=15, value=5)
with c3:
    forecast_years = st.number_input("Years to forecast ahead", min_value=1, max_value=20, value=5)
with c4:
    st.write("")
    st.write("")
    train_clicked = st.button("Train Model & Forecast", type="primary")

if train_clicked:
    with st.spinner(f"Training hybrid SARIMAX+LSTM for {country}… this can take up to a minute."):
        try:
            result = mp.train_and_forecast(
                df, country, test_years=int(test_years), forecast_years=int(forecast_years)
            )
            st.session_state.result = result
            st.success(
                f"Done — hybrid MAPE {result['metrics']['hybrid']['mape']:.2f}% "
                f"vs SARIMAX {result['metrics']['sarimax']['mape']:.2f}%."
            )
        except mp.InsufficientDataError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Training failed: {e}")

# --------------------------------------------------------------------------
# 3. Results
# --------------------------------------------------------------------------
result = st.session_state.result
if result is not None:
    st.header("3. Results")

    hist_years = result["history"]["years"]
    hist_vals = result["history"]["actual"]
    test_years_list = result["metrics"]["test_years"]
    sarimax_test = result["metrics"]["sarimax"]["predictions"]
    hybrid_test = result["metrics"]["hybrid"]["predictions"]
    fut_years = result["forecast"]["years"]
    fut_sarimax = result["forecast"]["sarimax"]
    fut_hybrid = result["forecast"]["hybrid"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hist_years, hist_vals, label="Actual (history)", color="#042C53", marker="o", markersize=3)
    ax.plot(test_years_list, sarimax_test, label="SARIMAX (test)", color="#B5D4F4", linestyle="--", marker="o", markersize=3)
    ax.plot(test_years_list, hybrid_test, label="Hybrid (test)", color="#1D9E75", marker="o", markersize=3)
    ax.plot(fut_years, fut_hybrid, label="Hybrid forecast", color="#E63946", marker="o", markersize=4)
    ax.plot(fut_years, fut_sarimax, label="SARIMAX forecast", color="#F4A261", linestyle="--", marker="o", markersize=3)
    ax.set_xlabel("Year")
    ax.set_ylabel("CO₂ emissions (kt)")
    ax.set_title(f"{country} — Hybrid SARIMAX+LSTM Forecast")
    ax.legend()
    ax.grid(alpha=0.25)
    st.pyplot(fig)

    m = result["metrics"]
    metrics_df = pd.DataFrame(
        {
            "Model": ["SARIMAX", "Hybrid (SARIMAX+LSTM)"],
            "MAPE (%)": [round(m["sarimax"]["mape"], 2), round(m["hybrid"]["mape"], 2)],
            "RMSE": [round(m["sarimax"]["rmse"]), round(m["hybrid"]["rmse"])],
            "R²": [round(m["sarimax"]["r2"], 3), round(m["hybrid"]["r2"], 3)],
        }
    )
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)

    with st.expander("Forecast values"):
        st.dataframe(
            pd.DataFrame(
                {"Year": fut_years, "SARIMAX forecast": fut_sarimax, "Hybrid forecast": fut_hybrid}
            ),
            hide_index=True,
            use_container_width=True,
        )
else:
    st.info("Pick a country and click **Train Model & Forecast** to see results.")
