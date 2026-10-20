"""
Page : Dashboard (executive summary)
"""

import pandas as pd
import plotly.express as px
import streamlit as st


@st.cache_data
def _load_dashboard_data():
    df = pd.read_csv("data/merged_retail_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def dashboard():
    "One-screen executive summary for stakeholders / recruiters."

    st.title("Dashboard")
    st.caption("Executive snapshot: scale, seasonality, store performance, holiday effect.")

    try:
        data = _load_dashboard_data()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    # ── KPIs ──────────────────────────────────────────────────────────────────
    total_revenue = data["Weekly_Sales"].sum()
    n_stores = int(data["Store"].nunique())
    n_depts = int(data["Dept"].nunique())
    holiday_col = "IsHoliday_x" if "IsHoliday_x" in data.columns else "IsHoliday"
    is_holiday = data[holiday_col].astype(bool)
    med_h = data.loc[is_holiday, "Weekly_Sales"].median()
    med_nh = data.loc[~is_holiday, "Weekly_Sales"].median()
    holiday_lift_pct = (med_h / med_nh - 1) * 100 if med_nh else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total revenue", f"${total_revenue / 1e9:.1f} B")
    col2.metric("Stores", f"{n_stores}")
    col3.metric("Departments", f"{n_depts}")
    col4.metric("Holiday median lift", f"+{holiday_lift_pct:.0f}%")

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

    st.subheader("Key takeaways")
    st.markdown(
        f"""
- **Seasonality** is the main time pattern: sharp peaks near Thanksgiving / Christmas.
- **Holiday weeks** show a higher median (+{holiday_lift_pct:.0f}% vs non-holiday).
- **Store concentration**: top locations drive a large share of total revenue.
- **Baseline for any forecast**: mean sales by (Store, Dept) already captures most structure
  (see Modeling annex if relevant).
        """
    )

    st.info(
        "Deeper charts → **Visualization**. Data quality & sources → **Exploration**. "
        "Forecasting experiments → **Modeling** (optional).",
        icon="👉",
    )
