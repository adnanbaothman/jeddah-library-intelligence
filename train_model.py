import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "jeddah_library_rentals.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def load_and_clean_data() -> pd.DataFrame:
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


def main():
    df = load_and_clean_data()
    target = "Rentals_Count"
    numeric_cols = [
        "Hour", "Temperature_C", "Humidity_pct", "Wind_Speed_ms", "Visibility_m",
        "Solar_Radiation_MJm2", "Rainfall_mm", "Snowfall_cm", "Month", "Day"
    ]
    categorical_cols = [
        "Season", "Holiday", "Functioning_Day", "Library_Branch",
        "Top_Category", "Membership_Type", "Day_of_Week"
    ]
    features = numeric_cols + categorical_cols
    X = df[features]
    y = df[target]
    mask = y.notna()
    X = X[mask]
    y = y[mask]

    preprocess = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
    ])

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=12),
        "Random Forest": RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1, min_samples_leaf=2),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = []
    pipelines = {}

    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", preprocess), ("model", estimator)])
        pipe.fit(X_train, y_train)
        predictions = pipe.predict(X_test)
        results.append({
            "Model": name,
            "RMSE": float(mean_squared_error(y_test, predictions) ** 0.5),
            "MAE": float(mean_absolute_error(y_test, predictions)),
            "R2": float(r2_score(y_test, predictions)),
        })
        pipelines[name] = pipe

    best_model_name = max(results, key=lambda item: item["R2"])["Model"]
    joblib.dump(pipelines[best_model_name], MODEL_DIR / "library_demand_model.pkl")

    metrics = {
        "results": results,
        "best_model": best_model_name,
        "rows": int(len(df)),
        "features": features,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
