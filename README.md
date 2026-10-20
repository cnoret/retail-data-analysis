# Walmart Retail Sales Analysis

[![Live Demo](https://img.shields.io/badge/Live%20App-Open-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://retail-data-analysis.streamlit.app/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cnoret/retail-data-analysis/blob/main/WalmartSalesPrediction.ipynb)
[![CI](https://github.com/cnoret/retail-data-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/cnoret/retail-data-analysis/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Analysis-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

End-to-end **retail sales analysis** on Walmart weekly data: exploration, cleaning, KPIs, interactive dashboards, and a lightweight forecasting annex. Built as a portfolio project for **Data Analyst** roles.

**Dataset:** 421,570 weekly records · 45 stores · 81 departments · Feb 2010 – Oct 2012 · $6.7B total revenue

**Live app:** [retail-data-analysis.streamlit.app](https://retail-data-analysis.streamlit.app/)

---

## Business insights

- Sales are dominated by **store × department identity**: a simple historical mean per (Store, Dept) already explains most of the variance. Any forecast must beat that bar.
- Clear **seasonality**: large spikes around Thanksgiving / Christmas, with a secondary post-holiday bump.
- **Holiday weeks** lift the median and widen the upper tail versus regular weeks.
- Store performance is uneven: a small set of stores (e.g. 20, 4, 14) concentrates a large share of revenue.
- Markdown promotions are sparse and only weakly related to sales in this dataset; economic indicators (CPI, unemployment) show mild negative association with sales.

---

## What’s in the app

| Page | Purpose |
| --- | --- |
| **Dashboard** | One-screen executive summary (KPIs, trend, top stores) |
| **Data Exploration** | Source tables, missing values, schema |
| **Data Processing** | Cleaning / merge walkthrough behind `merged_retail_data.csv` |
| **Visualization** | Correlations, distributions, rankings, holidays |
| **Modeling** | Optional forecasting annex (baseline vs simple models) |
| **Resources** | Stack and data source |

---

## Tech stack

| Layer | Tools |
| --- | --- |
| Analysis | Pandas, NumPy |
| Visualization | Plotly |
| App | Streamlit, streamlit-option-menu |
| Deployment | Streamlit Cloud |
| Annex | Scikit-Learn, Joblib (forecasting experiments only) |

---

## Quick start

```bash
git clone https://github.com/cnoret/retail-data-analysis.git
cd retail-data-analysis
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

App: `http://localhost:8501`

---

## Project structure

```text
retail-data-analysis/
├── app.py                         # Streamlit entry point
├── content/                       # App pages
├── data/                          # Raw CSVs + merged dataset
├── models/                        # Optional forecast artifacts
├── scripts/train_models.py        # Retrain helper (annex)
├── WalmartSalesPrediction.ipynb   # Notebook walkthrough
├── .github/workflows/ci.yml
└── requirements.txt
```

---

## Forecasting annex (optional)

A short modeling section tests whether ML adds value **beyond** a business baseline (mean sales per Store × Dept) on a **temporal** hold-out (cutoff `2012-04-13`).

| Approach | R² | RMSE | MAE |
| --- | --- | --- | --- |
| Residual model (baseline + RF on residuals) | ~0.96 | ~$4,500 | ~$2,300 |
| Baseline (Store × Dept mean) | ~0.94 | ~$5,300 | ~$2,600 |
| Direct Random Forest | ~0.88 | ~$7,600 | ~$4,500 |
| Linear Regression (foil) | ~0.06 | ~$21,000 | ~$15,000 |

**Analyst takeaway:** most signal is structural (which store / department). A residual model improves a bit on holidays and seasonality; it is not the core of this project.

Retrain (optional): `python scripts/train_models.py`

---

## License

MIT - [LICENSE](LICENSE).
