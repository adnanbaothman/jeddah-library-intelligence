import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Jeddah Library Intelligence",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "jeddah_library_rentals.csv"
MODEL_PATH = BASE_DIR / "models" / "library_demand_model.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"


# =========================
# Custom Styling
# =========================
st.markdown(
    """
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(37,99,235,0.22), transparent 30%),
            radial-gradient(circle at top right, rgba(124,58,237,0.16), transparent 28%),
            linear-gradient(180deg, #0B1020 0%, #111827 45%, #0F172A 100%);
        color: #E5E7EB;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        max-width: 1250px;
    }

    h1, h2, h3 {
        color: #F8FAFC !important;
        letter-spacing: -0.03em;
    }

    p, label, span, div {
        color: inherit;
    }

    .hero {
        padding: 34px 36px;
        border-radius: 28px;
        background:
            linear-gradient(135deg, rgba(37,99,235,0.96), rgba(79,70,229,0.92), rgba(15,23,42,0.98));
        border: 1px solid rgba(255,255,255,0.14);
        box-shadow: 0 24px 70px rgba(37,99,235,0.22);
        margin-bottom: 22px;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.65rem;
        line-height: 1.05;
    }

    .hero p {
        margin: 12px 0 0 0;
        color: #DBEAFE;
        font-size: 1.08rem;
        max-width: 780px;
    }

    .hero-badge {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,0.13);
        border: 1px solid rgba(255,255,255,0.20);
        color: #F8FAFC;
        font-size: 0.86rem;
        margin-bottom: 14px;
    }

    .metric-card {
        background: rgba(15,23,42,0.72);
        border: 1px solid rgba(148,163,184,0.22);
        border-radius: 22px;
        padding: 18px 18px;
        box-shadow: 0 16px 45px rgba(0,0,0,0.20);
        min-height: 112px;
    }

    .metric-label {
        color: #94A3B8;
        font-size: 0.88rem;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #F8FAFC;
        font-size: 1.85rem;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .metric-note {
        color: #93C5FD;
        font-size: 0.82rem;
    }

    .section-card {
        background: rgba(15,23,42,0.66);
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 24px;
        padding: 20px;
        margin: 12px 0 18px 0;
        box-shadow: 0 18px 55px rgba(0,0,0,0.20);
    }

    .insight-card {
        background: linear-gradient(180deg, rgba(30,41,59,0.82), rgba(15,23,42,0.82));
        border: 1px solid rgba(148,163,184,0.20);
        padding: 18px;
        border-radius: 20px;
        min-height: 145px;
        box-shadow: 0 12px 35px rgba(0,0,0,0.18);
    }

    .insight-card h3 {
        margin-top: 0;
        margin-bottom: 8px;
        font-size: 1.05rem;
    }

    .insight-card p {
        color: #CBD5E1;
        margin-bottom: 0;
        font-size: 0.94rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(15,23,42,0.68);
        border: 1px solid rgba(148,163,184,0.20);
        padding: 16px;
        border-radius: 18px;
    }

    div[data-testid="stTabs"] button {
        color: #CBD5E1;
        font-weight: 700;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #07111F 0%, #0F172A 100%);
        border-right: 1px solid rgba(148,163,184,0.16);
    }

    .footer {
        color: #94A3B8;
        text-align: center;
        padding-top: 22px;
        margin-top: 22px;
        border-top: 1px solid rgba(148,163,184,0.18);
        font-size: 0.9rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# Data Loading
# =========================
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="unicode_escape")
    df = df.drop_duplicates().copy()

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    num_cols = [
        "Hour",
        "Rentals_Count",
        "Temperature_C",
        "Humidity_pct",
        "Wind_Speed_ms",
        "Visibility_m",
        "Solar_Radiation_MJm2",
        "Rainfall_mm",
        "Snowfall_cm",
    ]

    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.loc[df["Temperature_C"] < 0, "Temperature_C"] = np.nan
    df.loc[df["Rentals_Count"] < 0, "Rentals_Count"] = np.nan

    weather_cols = [
        "Temperature_C",
        "Humidity_pct",
        "Wind_Speed_ms",
        "Visibility_m",
        "Solar_Radiation_MJm2",
        "Rainfall_mm",
    ]

    for col in weather_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    for col in ["Functioning_Day", "Holiday"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    df["Membership_Type"] = df["Membership_Type"].fillna("Unknown")
    df["Season"] = df["Season"].fillna("Unknown")

    df = df.dropna(subset=["Rentals_Count"])
    df = df[df["Functioning_Day"] == "Yes"].copy()

    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Is_Peak_Hour"] = df["Hour"].apply(lambda x: 1 if (9 <= x <= 11) or (16 <= x <= 19) else 0)
    df["Temperature_Bin"] = df["Temperature_C"].apply(
        lambda x: "Cool" if x < 25 else ("Warm" if x <= 35 else "Hot")
    )
    df["Is_Weekend"] = df["Date"].dt.weekday.apply(lambda x: 1 if x in [4, 5] else 0)

    return df.reset_index(drop=True)


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text())


def encode_for_model(row: dict, feature_columns: list[str]) -> pd.DataFrame:
    raw = pd.DataFrame([row])
    encoded = pd.get_dummies(raw, drop_first=True)
    return encoded.reindex(columns=feature_columns, fill_value=0)


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def demand_level(prediction: float, df: pd.DataFrame) -> tuple[str, str]:
    q25 = df["Rentals_Count"].quantile(0.25)
    q75 = df["Rentals_Count"].quantile(0.75)

    if prediction >= q75:
        return "High Demand", "Increase staffing and prepare high-demand categories."
    if prediction <= q25:
        return "Low Demand", "Good time for inventory checks and operational work."
    return "Moderate Demand", "Standard staffing should be enough."


def prediction_gauge(prediction: float, df: pd.DataFrame) -> go.Figure:
    max_value = max(float(df["Rentals_Count"].quantile(0.98)), prediction * 1.15, 1)
    q25 = float(df["Rentals_Count"].quantile(0.25))
    q75 = float(df["Rentals_Count"].quantile(0.75))

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(prediction),
            number={"suffix": " rentals", "font": {"size": 28}},
            title={"text": "Predicted Hourly Demand"},
            gauge={
                "axis": {"range": [0, max_value]},
                "bar": {"color": "#60A5FA"},
                "bgcolor": "rgba(15,23,42,0.2)",
                "borderwidth": 1,
                "bordercolor": "rgba(148,163,184,0.28)",
                "steps": [
                    {"range": [0, q25], "color": "rgba(34,197,94,0.28)"},
                    {"range": [q25, q75], "color": "rgba(250,204,21,0.28)"},
                    {"range": [q75, max_value], "color": "rgba(239,68,68,0.28)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=45, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E5E7EB"),
    )
    return fig


# =========================
# Load assets
# =========================
df = load_data()
artifact = load_artifact()
metrics = load_metrics()
results = pd.DataFrame(metrics["results"])


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.markdown("## 📚 Jeddah Library")
    st.caption("Smart operations dashboard")
    page = st.radio(
        "Navigation",
        ["Overview", "Demand Predictor", "Branch Analytics", "Model Comparison"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### Project Stack")
    st.write("Python · Streamlit · Scikit-learn · Plotly")
    st.markdown("### Data Scope")
    st.write(f"{df['Date'].min().date()} → {df['Date'].max().date()}")


# =========================
# Hero
# =========================
st.markdown(
    """
<div class="hero">
    <div class="hero-badge">Portfolio Project · Machine Learning Dashboard</div>
    <h1>📚 Jeddah Library Intelligence</h1>
    <p>
        A custom AI-powered analytics dashboard for forecasting library rental demand,
        understanding branch performance, and supporting smarter staffing decisions in Jeddah.
    </p>
</div>
""",
    unsafe_allow_html=True,
)


total_rentals = int(df["Rentals_Count"].sum())
avg_rentals = float(df["Rentals_Count"].mean())
branches = int(df["Library_Branch"].nunique())
peak_hour = int(df.groupby("Hour")["Rentals_Count"].mean().idxmax())
best_model_row = results.sort_values("R2", ascending=False).iloc[0]

k1, k2, k3, k4 = st.columns(4)
with k1:
    metric_card("Total Rentals", f"{total_rentals:,}", "Cleaned functioning-day records")
with k2:
    metric_card("Average Rentals", f"{avg_rentals:.1f}", "Mean hourly rental demand")
with k3:
    metric_card("Library Branches", f"{branches}", "Branches covered in dataset")
with k4:
    metric_card("Peak Hour", f"{peak_hour}:00", "Highest average demand")


# =========================
# Overview Page
# =========================
if page == "Overview":
    st.markdown("## Executive Summary")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="insight-card">
                <h3>🎯 Operational Goal</h3>
                <p>Predict hourly rental demand to support staffing, branch planning, and service readiness.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="insight-card">
                <h3>🏆 Best Notebook Model</h3>
                <p>{best_model_row["Model"]} achieved the strongest R² score based on the provided notebook metrics.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        busiest_branch = df.groupby("Library_Branch")["Rentals_Count"].sum().idxmax()
        st.markdown(
            f"""
            <div class="insight-card">
                <h3>📍 Busiest Branch</h3>
                <p>{busiest_branch} has the highest total rental activity in the cleaned dataset.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    c4, c5 = st.columns(2)

    hourly = df.groupby("Hour", as_index=False)["Rentals_Count"].mean()
    fig_hour = px.area(
        hourly,
        x="Hour",
        y="Rentals_Count",
        title="Average Demand Pattern Across the Day",
        markers=True,
    )
    fig_hour.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.35)")
    c4.plotly_chart(fig_hour, use_container_width=True)

    branch_total = (
        df.groupby("Library_Branch", as_index=False)["Rentals_Count"]
        .sum()
        .sort_values("Rentals_Count", ascending=True)
    )
    fig_branch = px.bar(
        branch_total,
        x="Rentals_Count",
        y="Library_Branch",
        orientation="h",
        title="Total Rentals by Branch",
        color="Rentals_Count",
        color_continuous_scale="Blues",
    )
    fig_branch.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.35)")
    c5.plotly_chart(fig_branch, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Demand Predictor Page
# =========================
elif page == "Demand Predictor":
    st.markdown("## Demand Predictor")
    st.caption("Simulate different operational scenarios and estimate expected hourly rental demand.")

    with st.container():
        c1, c2, c3 = st.columns(3)

        with c1:
            branch = st.selectbox("Library Branch", sorted(df["Library_Branch"].unique()))
            hour = st.slider("Hour", int(df["Hour"].min()), int(df["Hour"].max()), 17)
            day_name = st.selectbox(
                "Day of Week",
                ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
            )
            month = st.slider("Month", 1, 12, 3)

        with c2:
            temp = st.slider("Temperature °C", 15.0, 50.0, 32.0, 0.5)
            humidity = st.slider("Humidity %", 0.0, 100.0, 55.0, 1.0)
            wind = st.slider("Wind Speed m/s", 0.0, 10.0, 2.0, 0.1)
            visibility = st.slider("Visibility m", 0.0, 3000.0, 1800.0, 50.0)

        with c3:
            season = st.selectbox("Season", sorted(df["Season"].unique()))
            holiday = st.selectbox("Holiday", sorted(df["Holiday"].unique()))
            category = st.selectbox("Top Category", sorted(df["Top_Category"].unique()))
            membership = st.selectbox("Membership Type", sorted(df["Membership_Type"].unique()))

    input_row = {
        "Hour": hour,
        "Temperature_C": temp,
        "Humidity_pct": humidity,
        "Wind_Speed_ms": wind,
        "Visibility_m": visibility,
        "Solar_Radiation_MJm2": float(df["Solar_Radiation_MJm2"].median()),
        "Rainfall_mm": 0.0,
        "Season": season,
        "Holiday": holiday,
        "Functioning_Day": "Yes",
        "Library_Branch": branch,
        "Top_Category": category,
        "Membership_Type": membership,
        "Day_of_Week": day_name,
        "Month": month,
        "Day": 15,
        "Is_Peak_Hour": 1 if (9 <= hour <= 11) or (16 <= hour <= 19) else 0,
        "Temperature_Bin": "Cool" if temp < 25 else ("Warm" if temp <= 35 else "Hot"),
        "Is_Weekend": 1 if day_name in ["Friday", "Saturday"] else 0,
    }

    X = encode_for_model(input_row, artifact["feature_columns"])
    prediction = float(artifact["model"].predict(artifact["scaler"].transform(X))[0])
    level, recommendation = demand_level(prediction, df)

    r1, r2 = st.columns([1.15, 0.85])
    with r1:
        st.plotly_chart(prediction_gauge(prediction, df), use_container_width=True)

    with r2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.metric("Predicted Rentals", f"{prediction:,.0f}")
        st.metric("Demand Level", level)
        st.metric("Live Predictor", metrics["deployed_predictor"])
        st.markdown("</div>", unsafe_allow_html=True)

    if level == "High Demand":
        st.success(f"Recommendation: {recommendation}")
    elif level == "Low Demand":
        st.info(f"Recommendation: {recommendation}")
    else:
        st.warning(f"Recommendation: {recommendation}")

    report = pd.DataFrame(
        [
            {
                "Branch": branch,
                "Hour": hour,
                "Day": day_name,
                "Month": month,
                "Season": season,
                "Predicted_Rentals": round(prediction, 2),
                "Demand_Level": level,
                "Recommendation": recommendation,
            }
        ]
    )

    st.download_button(
        "Download Prediction Scenario",
        report.to_csv(index=False).encode("utf-8"),
        file_name="jeddah_library_prediction_scenario.csv",
        mime="text/csv",
    )


# =========================
# Branch Analytics Page
# =========================
elif page == "Branch Analytics":
    st.markdown("## Branch Analytics")
    st.caption("Explore demand patterns across branches, seasons, days, and membership types.")

    selected_branches = st.multiselect(
        "Filter branches",
        sorted(df["Library_Branch"].unique()),
        default=sorted(df["Library_Branch"].unique()),
    )

    filtered = df[df["Library_Branch"].isin(selected_branches)].copy()

    c1, c2 = st.columns(2)

    branch_total = (
        filtered.groupby("Library_Branch", as_index=False)["Rentals_Count"]
        .sum()
        .sort_values("Rentals_Count", ascending=True)
    )
    fig1 = px.bar(
        branch_total,
        x="Rentals_Count",
        y="Library_Branch",
        orientation="h",
        title="Total Rentals by Branch",
        color="Rentals_Count",
        color_continuous_scale="Blues",
    )
    fig1.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.35)")
    c1.plotly_chart(fig1, use_container_width=True)

    season_avg = (
        filtered.groupby("Season", as_index=False)["Rentals_Count"]
        .mean()
        .sort_values("Rentals_Count", ascending=False)
    )
    fig2 = px.bar(
        season_avg,
        x="Season",
        y="Rentals_Count",
        title="Average Rentals by Season",
        color="Rentals_Count",
        color_continuous_scale="Purples",
    )
    fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.35)")
    c2.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)

    hourly = filtered.groupby("Hour", as_index=False)["Rentals_Count"].mean()
    fig3 = px.line(hourly, x="Hour", y="Rentals_Count", markers=True, title="Average Rentals by Hour")
    fig3.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.35)")
    c3.plotly_chart(fig3, use_container_width=True)

    day_order = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    day_avg = filtered.groupby("Day_of_Week", as_index=False)["Rentals_Count"].mean()
    day_avg["Day_of_Week"] = pd.Categorical(day_avg["Day_of_Week"], day_order, ordered=True)
    day_avg = day_avg.sort_values("Day_of_Week")
    fig4 = px.bar(
        day_avg,
        x="Day_of_Week",
        y="Rentals_Count",
        title="Average Rentals by Day of Week",
        color="Rentals_Count",
        color_continuous_scale="Teal",
    )
    fig4.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.35)")
    c4.plotly_chart(fig4, use_container_width=True)

    st.markdown("### Business Insights")
    b1, b2, b3 = st.columns(3)

    current_peak_hour = int(filtered.groupby("Hour")["Rentals_Count"].mean().idxmax())
    top_category = filtered.groupby("Top_Category")["Rentals_Count"].sum().idxmax()
    top_member = filtered.groupby("Membership_Type")["Rentals_Count"].sum().idxmax()

    with b1:
        st.markdown(
            f"""
            <div class="insight-card">
                <h3>⏰ Staffing Window</h3>
                <p>The highest average demand occurs around {current_peak_hour}:00. This is the strongest candidate for extra staffing.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with b2:
        st.markdown(
            f"""
            <div class="insight-card">
                <h3>📖 Category Focus</h3>
                <p>{top_category} is the top rental category in the selected data. Prepare availability around this section.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with b3:
        st.markdown(
            f"""
            <div class="insight-card">
                <h3>👥 Membership Signal</h3>
                <p>{top_member} members generate the largest rental activity. This can guide membership campaigns.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    member = filtered["Membership_Type"].value_counts().reset_index()
    member.columns = ["Membership_Type", "Count"]
    fig5 = px.pie(
        member,
        names="Membership_Type",
        values="Count",
        title="Rentals by Membership Type",
        hole=0.52,
    )
    fig5.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig5, use_container_width=True)


# =========================
# Model Comparison Page
# =========================
elif page == "Model Comparison":
    st.markdown("## Model Comparison")
    st.caption("Model metrics are loaded directly from your project metrics file.")

    display = results.copy()
    display["R2"] = display["R2"].map(lambda x: f"{x:.6f}")
    display["MAE"] = display["MAE"].map(lambda x: f"{x:.6f}")
    display["RMSE"] = display["RMSE"].map(lambda x: f"{x:.6f}")

    st.dataframe(display, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)

    fig_r2 = px.bar(
        results.sort_values("R2", ascending=False),
        x="Model",
        y="R2",
        text="R2",
        title="R² Score by Model",
        color="R2",
        color_continuous_scale="Greens",
    )
    fig_r2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_r2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.35)")
    c1.plotly_chart(fig_r2, use_container_width=True)

    fig_mae = px.bar(
        results.sort_values("MAE"),
        x="Model",
        y="MAE",
        text="MAE",
        title="MAE by Model",
        color="MAE",
        color_continuous_scale="Reds_r",
    )
    fig_mae.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_mae.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.35)")
    c2.plotly_chart(fig_mae, use_container_width=True)

    deployed = metrics.get("deployed_predictor", "Deployed model")
    best_name = best_model_row["Model"]
    best_r2 = best_model_row["R2"]

    st.info(
        f"Best notebook model: {best_name} with R² = {best_r2:.6f}. "
        f"The deployed live predictor is {deployed}, selected for reliable Hugging Face deployment."
    )


st.markdown(
    """
<div class="footer">
Built as a portfolio-ready ML dashboard · Streamlit · Plotly · Scikit-learn
</div>
""",
    unsafe_allow_html=True,
)
