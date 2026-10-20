"""
Page : Dashboard (executive summary)
"""

import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

RESIDUAL_INFO = "models/residual_random_forest_info.json"
BASELINE_R2 = 0.9424  # Store×Dept mean on temporal hold-out (see Modeling)


@st.cache_data
def _load_dashboard_data():
    df = pd.read_csv("data/merged_retail_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data
def _load_residual_info():
    if not os.path.exists(RESIDUAL_INFO):
        return None
    with open(RESIDUAL_INFO, encoding="utf-8") as f:
        return json.load(f)


def dashboard():
    "One-screen executive summary for recruiters."

    st.title("Dashboard")
    st.caption("Executive snapshot: data scale, sales pulse, and the model that beats the baseline.")

    try:
        data = _load_dashboard_data()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    info = _load_residual_info()

    # ── KPIs ──────────────────────────────────────────────────────────────────
    total_revenue = data["Weekly_Sales"].sum()
    n_stores = data["Store"].nunique()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total revenue", f"${total_revenue / 1e9:.1f} B")
    col2.metric("Stores", f"{n_stores}")
    if info:
        col3.metric("Residual RF R²", f"{info['r2']:.2f}")
        col4.metric("Residual RF RMSE", f"${info['rmse']:,.0f}")
    else:
        col3.metric("Residual RF R²", "n/a")
        col4.metric("Residual RF RMSE", "n/a")

    st.write("---")

    # ── Charts ────────────────────────────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.subheader("Sales trend")
        trend = (
            data.groupby("Date", as_index=False)["Weekly_Sales"]
            .sum()
            .sort_values("Date")
        )
        fig = px.line(
            trend,
            x="Date",
            y="Weekly_Sales",
            labels={"Weekly_Sales": "Total sales ($)", "Date": ""},
        )
        fig.update_traces(line_color="#0071CE")
        fig.update_layout(height=340, margin={"t": 10, "b": 10, "l": 10, "r": 10})
        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("Top 10 stores")
        top = (
            data.groupby("Store", as_index=False)["Weekly_Sales"]
            .sum()
            .nlargest(10, "Weekly_Sales")
        )
        top["Store"] = top["Store"].astype(str)
        fig = px.bar(
            top,
            x="Weekly_Sales",
            y="Store",
            orientation="h",
            labels={"Weekly_Sales": "Total sales ($)", "Store": "Store"},
            color="Weekly_Sales",
            color_continuous_scale="Blues",
        )
        fig.update_layout(
            height=340,
            margin={"t": 10, "b": 10, "l": 10, "r": 10},
            coloraxis_showscale=False,
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig, width="stretch")

    st.write("---")

    # ── Model takeaway + CTA ──────────────────────────────────────────────────
    if info:
        delta = info["r2"] - BASELINE_R2
        st.success(
            f"**Residual Random Forest** beats the Store×Dept mean baseline on a temporal "
            f"hold-out (cutoff {info['cutoff_date']}): "
            f"R² **{info['r2']:.4f}** vs ~{BASELINE_R2:.2f} (Δ +{delta:.3f}), "
            f"RMSE **${info['rmse']:,.0f}**.",
            icon="🏆",
        )
    else:
        st.info("Train models via `python scripts/train_models.py` to populate model KPIs.")

    st.info(
        "Details and live forecasts are on **Modeling**. "
        "Deeper charts are on **Visualization**.",
        icon="👉",
    )
