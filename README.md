# Walmart Sales Forecasting

[![Live Demo](https://img.shields.io/badge/Live%20App-Open-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://retail-data-analysis.streamlit.app/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cnoret/retail-data-analysis/blob/main/WalmartSalesPrediction.ipynb)
[![CI](https://github.com/cnoret/retail-data-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/cnoret/retail-data-analysis/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Predicting weekly department-level sales across 45 Walmart stores using machine learning, built as an end-to-end data science project with an interactive Streamlit app.

**Dataset:** 421,570 weekly records · 45 stores · 81 departments · Feb 2010 – Oct 2012 · $6.7B total revenue

---

## Features

- **Data Exploration**: interactive overview of the three source datasets with missing value analysis
- **Data Processing**: cleaning pipeline: imputation, date parsing, and dataset merging
- **Analysis & Visualization**: interactive Plotly charts: correlation matrix, sales distribution, store rankings, time trends, and holiday impact
- **Modeling**: temporal hold-out, calendar features (`Week`/`Month`), Store×Dept baseline, Linear Regression (pedagogical foil) vs Random Forest, with R² / RMSE / MAE, Actual vs Predicted, and feature importance
- **Live Predictions**: input any store/department/context and get an instant sales forecast

## Tech Stack

| Layer | Libraries |
| --- | --- |
| Data | Pandas, NumPy |
| ML | Scikit-Learn (LinearRegression, RandomForestRegressor) · Joblib (model persistence) |
| Visualization | Plotly |
| App | Streamlit |
| Deployment | Streamlit Cloud |

## Quick Start

```bash
git clone https://github.com/cnoret/retail-data-analysis.git
cd retail-data-analysis
pip install -r requirements.txt
streamlit run app.py
```

App available at `http://localhost:8501`

## Project Structure

```text
retail-data-analysis/
├── app.py                  # Entry point
├── content/
│   ├── intro.py
│   ├── exploration.py
│   ├── preparation.py
│   ├── visualisation.py
│   ├── modelisation.py
│   └── resources.py
├── data/                   # CSV datasets
├── models/                 # Pre-trained models (joblib)
├── images/                 # UI assets
└── requirements.txt
```

## Results

Evaluation uses a **temporal 80/20 split** on unique weeks (train before `2012-04-13`, test after), not a random split. Features include store/department context, holiday flag, economic indicators, store type, and calendar fields (`Week`, `Month`).

| Model | R² | RMSE | MAE |
| --- | --- | --- | --- |
| Baseline (Store × Dept mean) | ~0.94 | ~$5,300 | ~$2,600 |
| Random Forest (`n=20`, `depth=10`) | ~0.87 | ~$7,800 | ~$4,600 |
| Linear Regression | ~0.06 | ~$21,000 | ~$15,000 |

**Takeaways**
- Linear Regression is kept as a **pedagogical foil**: raw numeric `Store`/`Dept` IDs are not linear features, so it collapses.
- Random Forest handles those IDs via splits and beats LR, but **does not beat the Store×Dept mean baseline** on this hold-out. Most signal is identity, not exogenous covariates.
- RF hyperparameters stay light for Streamlit Cloud memory limits. Next gains need lags or residual modeling on top of the baseline.

## License

MIT - [LICENSE](LICENSE).
