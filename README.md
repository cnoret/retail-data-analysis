# Walmart Store Performance Review

[![Live Demo](https://img.shields.io/badge/Live%20App-Open-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://retail-data-analysis.streamlit.app/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cnoret/retail-data-analysis/blob/main/WalmartSalesPrediction.ipynb)
[![CI](https://github.com/cnoret/retail-data-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/cnoret/retail-data-analysis/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Analysis-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**Regional performance review** of 45 Walmart stores: where revenue is lost, which stores and departments need attention, and how to plan the holiday peak. Delivered as an interactive Streamlit app backed by reproducible pandas analysis.

**Data:** 421,570 weekly sales records · 45 stores · 81 departments · Feb 2010 – Oct 2012 · $6.7B revenue (~$2.45B a year)

**Live app:** [retail-data-analysis.streamlit.app](https://retail-data-analysis.streamlit.app/)

---

## The brief

> *A regional manager overseeing 45 stores needs to set priorities for next year: which stores underperform, which departments to back, how to prepare for the holiday season, and whether markdown promotions are worth their cost.*

This project answers those four questions with the chain's own sales history, then turns the answers into actions.

---

## Recommendations

1. **Audit 7 underperforming stores first.** Stores 9, 15, 21, 25, 29, 45 (type B) and 44 (type C) sell **25–51% less per square foot** than the median store of their type. Bringing them to that median would be worth **up to ~$140M a year** (about 6% of revenue).
2. **Investigate the decline of large stores before it spreads.** 18 of 45 stores sold less in Feb–Oct 2012 than in the same months of 2010. Store 14, the #3 store by revenue, lost **15%**, and stores 35 and 36 lost 22% and 32%.
3. **Plan the holiday peak on the real calendar, not on the dataset's holiday flag.** The week before Christmas sells **+67–75%** above a median week and Thanksgiving week **+42–44%**. The flagged "Christmas" week (the one ending after Dec 25) sells **at or below** a median week. Nov–Dec represents **~20% of yearly revenue in 17% of the weeks** (2011). Each `Date` is the Friday that ends the week.
4. **Back the departments that carry growth, review the ones that shrink.** The top 10 departments make **46% of revenue**. Departments 92, 87 and 90 drove most of the growth (+$26M between Feb–Oct 2010 and 2012), while departments 55 and 6 fell by 25% and 35%.
5. **Test markdowns before scaling them.** $40M of markdowns were recorded from Nov 2011, with no measurable sales lift: outside holidays, a store sells the same in its high-markdown weeks as in its other weeks (−0.1% vs its average week). A controlled test (matched stores, with and without markdowns) is needed before spending more.

---

## Key findings

| Question | Finding | Evidence |
| --- | --- | --- |
| Where is revenue concentrated? | Type A stores make 64% of revenue; the top 10 stores make 39% | Store ranking, sales per sq ft by type |
| Which stores underperform? | 7 stores sit more than 25% below their type's median sales per sq ft | Size-adjusted store comparison |
| Is the business growing? | Flat overall (+1.9% Feb–Oct 2012 vs 2010), with 18 stores in decline | Year-over-year on comparable months |
| How strong is seasonality? | December weeks sell 25% above other weeks; the most seasonal departments (7, 55, 5) sell 2.4–3.4× more | Weekly trend, department uplift |
| Do external factors matter? | CPI, unemployment, fuel price and temperature show near-zero correlation with sales (\|r\| < 0.03) | Correlation analysis |

Store and department identity explain most of the variation: a simple average per (store, department) already captures ~90% of it. Performance is mostly a matter of **which store and which department**, which is why the recommendations target specific stores and departments.

---

## What’s in the app

| Page | Purpose |
| --- | --- |
| **Overview** | The brief and the five recommendations, with live figures |
| **Dashboard** | One-screen executive summary (KPIs, trend, top stores) |
| **Data Exploration** | Source tables, missing values, schema |
| **Data Processing** | Cleaning / merge walkthrough behind `merged_retail_data.csv` |
| **Visualization** | Correlations, distributions, rankings, holidays |
| **Modeling** | Optional forecasting annex (baseline vs simple models) |
| **Resources** | Stack and data source |

---

## SQL analysis

Each business question also has a plain SQL query in [`sql/`](sql/), run with [DuckDB](https://duckdb.org/) directly on the raw CSV files (no database server needed).

| File | Question |
| --- | --- |
| `00_staging.sql` | Typed views over the raw CSVs (`stores`, `sales`, `features`) |
| `01_store_productivity.sql` | Which stores underperform once size is taken into account? |
| `02_store_growth.sql` | Which stores grow or decline (Feb–Oct 2012 vs 2010)? |
| `03_holiday_peak.sql` | How big is the holiday peak vs a median week? |
| `04_department_mix.sql` | Which departments weigh the most, and how do they evolve? |
| `05_markdowns.sql` | Do markdown weeks sell more than other weeks? |

```bash
python scripts/run_sql.py          # run every query
python scripts/run_sql.py 03       # run one query
python scripts/run_sql.py --check  # check SQL and the app give the same figures (run in CI)
```

---

## Tech stack

| Layer | Tools |
| --- | --- |
| Analysis | Pandas, NumPy, SQL (DuckDB) |
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
├── sql/                           # SQL version of each business question (DuckDB)
├── scripts/run_sql.py             # Run the SQL queries / cross-check with the app
├── scripts/train_models.py        # Retrain helper (annex)
├── WalmartSalesPrediction.ipynb   # Notebook walkthrough
├── .github/workflows/ci.yml
└── requirements.txt
```

---

## Appendix: forecasting check

A short modeling section tests whether ML adds value **beyond** a business baseline (mean sales per Store × Dept) on a **temporal** hold-out (cutoff `2012-04-13`).

| Approach | R² | RMSE | MAE |
| --- | --- | --- | --- |
| Residual model (baseline + RF on residuals) | ~0.96 | ~$4,500 | ~$2,300 |
| Baseline (Store × Dept mean) | ~0.94 | ~$5,300 | ~$2,600 |
| Direct Random Forest | ~0.88 | ~$7,600 | ~$4,500 |
| Linear Regression (foil) | ~0.06 | ~$21,000 | ~$15,000 |

**Takeaway:** most of the signal is structural (which store, which department). A model on top of the baseline cuts the error a little (RMSE −15%), but forecasting is not the core of this review.

**Limit of this check:** the hold-out runs from April to October 2012, so it contains no Thanksgiving or Christmas week. It says nothing about how well the models forecast the holiday peak.

Retrain (optional): `python scripts/train_models.py`

---

## License

MIT - [LICENSE](LICENSE).
