# Walmart Sales Forecasting

[![Live Demo](https://img.shields.io/badge/Live%20App-Open-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://retail-data-analysis.streamlit.app/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cnoret/retail-data-analysis/blob/main/WalmartSalesPrediction.ipynb)
[![CI](https://github.com/cnoret/retail-data-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/cnoret/retail-data-analysis/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Predicting weekly department-level sales across 45 Walmart stores using machine learning, built as an end-to-end data science project with an interactive Streamlit app.

**Dataset:** 421,570 weekly records · 45 stores · 81 departments · Feb 2010 – Oct 2012 · $6.7B total revenue

---

## Features

- **Data Exploration**: interactive overview of the three source datasets with missing value analysis
- **Data Processing**: walkthrough of the cleaning/merge pipeline that produced the committed merged CSV
- **Analysis & Visualization**: interactive Plotly charts (correlation, sales distribution, store rankings, trends, holidays)
- **Modeling**: temporal hold-out, calendar features, Store×Dept baseline, Linear Regression (foil), Random Forest, and **Residual Random Forest** (baseline + RF on residuals)
- **Live Predictions**: forecast for a real store/department and week date

## Tech Stack

| Layer | Libraries |
| --- | --- |
| Data | Pandas, NumPy |
| ML | Scikit-Learn (LinearRegression, RandomForestRegressor) · Joblib |
| Visualization | Plotly |
| App | Streamlit · streamlit-option-menu |
| Deployment | Streamlit Cloud |

## Quick Start

```bash
git clone https://github.com/cnoret/retail-data-analysis.git
cd retail-data-analysis
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

App available at `http://localhost:8501`

### Retrain models

```bash
python scripts/train_models.py
```

Writes joblib artifacts and `*_info.json` under `models/`.

## Project Structure

```text
retail-data-analysis/
├── app.py                         # Streamlit entry point
├── content/                       # App pages
├── data/                          # Raw CSVs + merged_retail_data.csv
├── models/                        # Pre-trained joblib artifacts
├── scripts/train_models.py        # Retrain helper
├── WalmartSalesPrediction.ipynb   # Colab / notebook walkthrough
├── .github/workflows/ci.yml
├── .streamlit/config.toml
└── requirements.txt
```

## Results

Evaluation uses a **temporal 80/20 split** on unique weeks (train before `2012-04-13`, test after). Negative `Weekly_Sales` rows (returns/adjustments) are dropped. Features: store/department, holiday flag, economic indicators, store type, `Week`, `Month`.

| Model | R² | RMSE | MAE |
| --- | --- | --- | --- |
| **Residual Random Forest** (baseline + RF) | **~0.96** | **~$4,500** | **~$2,300** |
| Baseline (Store × Dept mean) | ~0.94 | ~$5,300 | ~$2,600 |
| Random Forest (`n=20`, `depth=10`) | ~0.88 | ~$7,600 | ~$4,500 |
| Linear Regression | ~0.06 | ~$21,000 | ~$15,000 |

**Takeaways**
- Linear Regression is a **pedagogical foil**: raw numeric `Store`/`Dept` IDs are not linear features.
- A direct Random Forest beats LR but **loses to the Store×Dept mean** on this hold-out.
- **Residual Random Forest** keeps the strong identity baseline and models seasonal/contextual residuals, beating the baseline on R², RMSE, and MAE.
- RF hyperparameters stay light for Streamlit Cloud memory limits.

## License

MIT - [LICENSE](LICENSE).
