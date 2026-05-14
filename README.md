---
title: Jeddah Library Intelligence
emoji: 📚
colorFrom: orange
colorTo: blue
sdk: streamlit
sdk_version: "1.38.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

## Jeddah Library Intelligence

A professional machine learning dashboard that predicts hourly book rental demand across Jeddah public library branches using weather, time, membership, category, and operational factors.

## Live Demo

Deploy this repository as a Hugging Face Space using the **Streamlit** SDK.

## Business Problem

Public library branches need to plan staffing, inventory, and operating capacity based on expected hourly rental demand. This project predicts demand and turns raw data into decision-ready operational insights.

## Key Features

- Interactive hourly rental demand predictor.
- Branch-level demand analytics.
- Hour and day-of-week heatmap.
- Model comparison across Linear Regression, Decision Tree, Random Forest, and Gradient Boosting.
- Data explorer with CSV export.
- Clean production-style project structure for GitHub and Hugging Face Spaces.

## Model Results

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Linear Regression | 12.68 | 10.15 | 0.675 |
| Decision Tree | 9.83 | 7.00 | 0.805 |
| Random Forest | 6.95 | 4.99 | 0.902 |
| Gradient Boosting | 7.01 | 5.02 | 0.901 |

The production model is **Random Forest Regressor** because it achieved the strongest R² score and lowest error on the test set.

## Project Structure

```text
.
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── jeddah_library_rentals.csv
└── models/
    ├── library_demand_model.pkl
    └── metrics.json
```

## Run Locally

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Tech Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- Plotly
- Joblib

## Portfolio Summary

Jeddah Library Intelligence is an end-to-end machine learning application that converts a notebook experiment into a deployed decision-support dashboard. It demonstrates data cleaning, feature engineering, model comparison, explainability-oriented visuals, and deployment readiness.
