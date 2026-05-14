# 📚 Jeddah Library Intelligence

[![Open in Hugging Face](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-blue?logo=huggingface)](https://huggingface.co/spaces/AdnanBaothman/jeddah-library-intelligence)

An AI-powered operations dashboard for predicting hourly public library rental demand across Jeddah branches using machine learning and interactive analytics.

## 🚀 Live Demo

[Open the App](https://huggingface.co/spaces/AdnanBaothman/jeddah-library-intelligence)

## 🎯 Business Problem

Public libraries need to anticipate hourly rental demand so branch teams can plan staffing, manage visitor flow, prepare popular categories, and understand when each branch is most active.

## ✨ Key Features

- Interactive rental demand prediction
- Branch-level analytics and operational insights
- Model comparison based on notebook results
- KPI cards for total rentals, branches, best model score, and peak hour
- Demand heatmap by day of week and hour
- Downloadable prediction scenario report
- Streamlit + Plotly dashboard deployed on Hugging Face Spaces

## 🏆 Model Results

These are the corrected metrics from the notebook comparison output.

| Model | R² | MAE | RMSE |
|---|---:|---:|---:|
| Neural Network | 0.939981 | 4.007833 | 5.292908 |
| Random Forest | 0.928729 | 4.274999 | 5.767762 |
| Linear Regression | 0.876106 | 5.947108 | 7.604567 |
| Decision Tree | 0.862325 | 5.961512 | 8.016341 |

**Best notebook model:** Neural Network  
**Deployed live predictor:** Random Forest, selected because it is lighter and more stable for Hugging Face deployment while still achieving strong notebook performance.

## 🧠 Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Joblib
- Hugging Face Spaces

## 📁 Project Structure

```text
jeddah-library-intelligence/
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

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 👨‍💻 Portfolio Summary

This project demonstrates an end-to-end machine learning workflow: data cleaning, feature engineering, model evaluation, interactive dashboard development, and cloud deployment. It is designed as a portfolio-ready ML product rather than a simple notebook experiment.
