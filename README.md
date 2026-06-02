# Walmart Sales Forecasting

[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://retail-data-analysis.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Predicting weekly department-level sales across 45 Walmart stores using machine learning — built as an end-to-end data science project with an interactive Streamlit app.

**Dataset:** 421,570 weekly records · 45 stores · 81 departments · Feb 2010 – Oct 2012 · $6.7B total revenue

---

## Features

- **Data Exploration** — interactive overview of the three source datasets with missing value analysis
- **Data Processing** — cleaning pipeline: imputation, date parsing, and dataset merging
- **Analysis & Visualization** — interactive Plotly charts: correlation matrix, sales distribution, store rankings, time trends, and holiday impact
- **Modeling** — Linear Regression vs Random Forest with R², RMSE, MAE metrics, Actual vs Predicted chart, and feature importance
- **Live Predictions** — input any store/department/context and get an instant sales forecast

## Tech Stack

| Layer | Libraries |
| --- | --- |
| Data | Pandas, NumPy |
| ML | Scikit-Learn (LinearRegression, RandomForestRegressor) |
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
├── images/                 # UI assets
└── requirements.txt
```

## Results

| Model | R² | RMSE |
| --- | --- | --- |
| Linear Regression | ~0.06 | ~$22,000 |
| Random Forest | ~0.97 | ~$4,000 |

Random Forest significantly outperforms Linear Regression because `Store` and `Dept` are categorical identifiers — tree-based splits handle them naturally while linear models treat them as continuous values.

## License

MIT — see [LICENSE](LICENSE).
