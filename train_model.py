import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "jeddah_library_rentals.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

NOTEBOOK_RESULTS = [
    {"Model": "Linear Regression", "R2": 0.876106, "MAE": 5.947108, "RMSE": 7.604567},
    {"Model": "Decision Tree", "R2": 0.862325, "MAE": 5.961512, "RMSE": 8.016341},
    {"Model": "Random Forest", "R2": 0.928729, "MAE": 4.274999, "RMSE": 5.767762},
    {"Model": "Neural Network", "R2": 0.939981, "MAE": 4.007833, "RMSE": 5.292908},
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="unicode_escape")
    df = df.drop_duplicates().copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    num_cols = [
        "Hour", "Rentals_Count", "Temperature_C", "Humidity_pct", "Wind_Speed_ms",
        "Visibility_m", "Solar_Radiation_MJm2", "Rainfall_mm", "Snowfall_cm"
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.loc[df["Temperature_C"] < 0, "Temperature_C"] = np.nan
    df.loc[df["Rentals_Count"] < 0, "Rentals_Count"] = np.nan

    weather_cols = ["Temperature_C", "Humidity_pct", "Wind_Speed_ms", "Visibility_m", "Solar_Radiation_MJm2", "Rainfall_mm"]
    for col in weather_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in ["Functioning_Day", "Holiday"]:
        df[col] = df[col].fillna(df[col].mode()[0])

    df["Membership_Type"] = df["Membership_Type"].fillna("Unknown")
    df["Season"] = df["Season"].fillna("Unknown")
    df = df.dropna(subset=["Rentals_Count"])
    df = df[df["Functioning_Day"] == "Yes"].copy()

    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Is_Peak_Hour"] = df["Hour"].apply(lambda x: 1 if (9 <= x <= 11) or (16 <= x <= 19) else 0)
    df["Temperature_Bin"] = df["Temperature_C"].apply(lambda x: "Cool" if x < 25 else ("Warm" if x <= 35 else "Hot"))
    df["Is_Weekend"] = df["Date"].dt.weekday.apply(lambda x: 1 if x in [4, 5] else 0)
    return df.reset_index(drop=True)


def main() -> None:
    df = load_data()
    target = "Rentals_Count"
    feature_cols = [
        "Hour", "Temperature_C", "Humidity_pct", "Wind_Speed_ms", "Visibility_m",
        "Solar_Radiation_MJm2", "Rainfall_mm", "Season", "Holiday", "Functioning_Day",
        "Library_Branch", "Top_Category", "Membership_Type", "Day_of_Week", "Month", "Day",
        "Is_Peak_Hour", "Temperature_Bin", "Is_Weekend"
    ]

    X = pd.get_dummies(df[feature_cols], drop_first=True)
    y = df[target]
    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1, min_samples_leaf=2)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    live_metrics = {
        "R2": float(r2_score(y_test, preds)),
        "MAE": float(mean_absolute_error(y_test, preds)),
        "RMSE": float(mean_squared_error(y_test, preds) ** 0.5),
    }

    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_columns,
    }
    joblib.dump(artifact, MODEL_DIR / "library_demand_model.pkl")

    metrics = {
        "results": NOTEBOOK_RESULTS,
        "best_model": "Neural Network",
        "deployed_predictor": "Random Forest",
        "deployed_predictor_r2": 0.928729,
        "live_retrain_metrics": live_metrics,
        "rows": int(len(df)),
        "note": "Displayed model-comparison metrics are copied from the provided notebook output. Random Forest is deployed for lightweight Streamlit inference.",
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
