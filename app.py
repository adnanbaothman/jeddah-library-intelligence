import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "jeddah_library_rentals.csv"
MODEL_PATH = BASE_DIR / "models" / "library_demand_model.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"

st.set_page_config(
    page_title="Jeddah Library Intelligence",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="unicode_escape")
    df = df.drop_duplicates().copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    numeric_cols = [
        "Hour", "Rentals_Count", "Temperature_C", "Humidity_pct", "Wind_Speed_ms",
        "Visibility_m", "Solar_Radiation_MJm2", "Rainfall_mm", "Snowfall_cm", "Month", "Day"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["Temperature_C", "Rentals_Count"]:
        df.loc[df[col] < 0, col] = np.nan
    return df

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {}


def prepare_input(row: dict) -> pd.DataFrame:
    return pd.DataFrame([row])


df = load_data()
model = load_model()
metrics = load_metrics()

st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Choose a section",
    ["Demand Predictor", "Business Insights", "Branch Analytics", "Model Performance", "Data Explorer"],
)

st.markdown(
    """
    <div style="padding: 1.4rem; border-radius: 1rem; background: linear-gradient(90deg, #f6eadf, #eef6ff); margin-bottom: 1rem;">
        <h1 style="margin:0; color:#1f2937;">📚 Jeddah Library Intelligence</h1>
        <p style="margin-top:.5rem; color:#374151; font-size:1.05rem;">
            A machine learning dashboard for predicting hourly book rental demand across Jeddah public library branches.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if section == "Demand Predictor":
    st.subheader("Predict Hourly Library Rentals")
    col1, col2, col3 = st.columns(3)

    with col1:
        hour = st.slider("Hour", 0, 23, 14)
        month = st.slider("Month", 1, 12, 3)
        day = st.slider("Day of Month", 1, 31, 15)
        temperature = st.slider("Temperature (°C)", 15.0, 50.0, 32.0, 0.5)
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 55.0, 1.0)
    with col2:
        wind = st.slider("Wind Speed (m/s)", 0.0, 10.0, 2.0, 0.1)
        visibility = st.slider("Visibility (m)", 0.0, 3000.0, 1800.0, 50.0)
        solar = st.slider("Solar Radiation", 0.0, 5.0, 1.5, 0.1)
        rainfall = st.slider("Rainfall (mm)", 0.0, 20.0, 0.0, 0.1)
        snowfall = st.slider("Snowfall (cm)", 0.0, 5.0, 0.0, 0.1)
    with col3:
        season = st.selectbox("Season", sorted(df["Season"].dropna().unique()))
        holiday = st.selectbox("Holiday", sorted(df["Holiday"].dropna().unique()))
        functioning_day = st.selectbox("Functioning Day", sorted(df["Functioning_Day"].dropna().unique()))
        branch = st.selectbox("Library Branch", sorted(df["Library_Branch"].dropna().unique()))
        category = st.selectbox("Top Category", sorted(df["Top_Category"].dropna().unique()))
        membership = st.selectbox("Membership Type", sorted(df["Membership_Type"].dropna().unique()))
        day_of_week = st.selectbox("Day of Week", sorted(df["Day_of_Week"].dropna().unique()))

    input_row = {
        "Hour": hour,
        "Temperature_C": temperature,
        "Humidity_pct": humidity,
        "Wind_Speed_ms": wind,
        "Visibility_m": visibility,
        "Solar_Radiation_MJm2": solar,
        "Rainfall_mm": rainfall,
        "Snowfall_cm": snowfall,
        "Month": month,
        "Day": day,
        "Season": season,
        "Holiday": holiday,
        "Functioning_Day": functioning_day,
        "Library_Branch": branch,
        "Top_Category": category,
        "Membership_Type": membership,
        "Day_of_Week": day_of_week,
    }

    prediction = model.predict(prepare_input(input_row))[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Rentals", f"{prediction:,.0f}")
    c2.metric("Selected Branch", branch)
    c3.metric("Best Model", metrics.get("best_model", "Random Forest"))

    if prediction >= df["Rentals_Count"].quantile(0.75):
        st.success("High demand expected. Consider increasing staffing and preparing high-demand categories.")
    elif prediction <= df["Rentals_Count"].quantile(0.25):
        st.info("Low demand expected. This is a good time for inventory work or staff training.")
    else:
        st.warning("Moderate demand expected. Standard staffing should be sufficient.")

elif section == "Business Insights":
    st.subheader("Business Insights")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Records", f"{len(df):,}")
    k2.metric("Avg. Rentals", f"{df['Rentals_Count'].mean():.1f}")
    k3.metric("Branches", df["Library_Branch"].nunique())
    k4.metric("Categories", df["Top_Category"].nunique())

    hourly = df.groupby("Hour", as_index=False)["Rentals_Count"].mean()
    fig = px.line(hourly, x="Hour", y="Rentals_Count", markers=True, title="Average Rentals by Hour")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        season_df = df.groupby("Season", as_index=False)["Rentals_Count"].mean().sort_values("Rentals_Count", ascending=False)
        st.plotly_chart(px.bar(season_df, x="Season", y="Rentals_Count", title="Average Rentals by Season"), use_container_width=True)
    with c2:
        cat_df = df.groupby("Top_Category", as_index=False)["Rentals_Count"].mean().sort_values("Rentals_Count", ascending=False)
        st.plotly_chart(px.bar(cat_df, x="Top_Category", y="Rentals_Count", title="Average Rentals by Book Category"), use_container_width=True)

elif section == "Branch Analytics":
    st.subheader("Branch Analytics")
    selected = st.multiselect("Select branches", sorted(df["Library_Branch"].dropna().unique()), default=sorted(df["Library_Branch"].dropna().unique())[:3])
    view = df[df["Library_Branch"].isin(selected)] if selected else df

    branch = view.groupby("Library_Branch", as_index=False).agg(
        Average_Rentals=("Rentals_Count", "mean"),
        Total_Rentals=("Rentals_Count", "sum"),
        Records=("Rentals_Count", "count"),
    ).sort_values("Average_Rentals", ascending=False)
    st.dataframe(branch, use_container_width=True)
    st.plotly_chart(px.bar(branch, x="Library_Branch", y="Average_Rentals", title="Average Demand by Branch"), use_container_width=True)

    heat = view.pivot_table(index="Day_of_Week", columns="Hour", values="Rentals_Count", aggfunc="mean")
    st.plotly_chart(px.imshow(heat, aspect="auto", title="Demand Heatmap: Day of Week vs Hour"), use_container_width=True)

elif section == "Model Performance":
    st.subheader("Model Performance")
    results = pd.DataFrame(metrics.get("results", []))
    if not results.empty:
        results["R2"] = results["R2"].round(3)
        results["RMSE"] = results["RMSE"].round(2)
        results["MAE"] = results["MAE"].round(2)
        st.dataframe(results, use_container_width=True)
        st.plotly_chart(px.bar(results, x="Model", y="R2", title="Model Comparison by R² Score"), use_container_width=True)
        st.caption(f"Production model: {metrics.get('best_model', 'Random Forest')}")
    else:
        st.warning("Model metrics were not found.")

elif section == "Data Explorer":
    st.subheader("Data Explorer")
    branches = st.multiselect("Filter by branch", sorted(df["Library_Branch"].dropna().unique()))
    filtered = df[df["Library_Branch"].isin(branches)] if branches else df
    st.dataframe(filtered, use_container_width=True)
    st.download_button(
        "Download filtered data as CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        "filtered_jeddah_library_rentals.csv",
        "text/csv",
    )
