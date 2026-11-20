"""
Page : Dashboard (executive summary)
"""

import plotly.express as px
import streamlit as st

from content.data import load_merged
from content.intro import compute_headline_figures


def _load_dashboard_data():
    return load_merged()


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
    figures = compute_headline_figures(data)
    store_totals = data.groupby("Store")["Weekly_Sales"].sum()
    top10_store_share = store_totals.nlargest(10).sum() / store_totals.sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total revenue", f"${total_revenue / 1e9:.1f} B")
    col2.metric("Stores", f"{n_stores}")
    col3.metric("Departments", f"{n_depts}")
    col4.metric(
        "Peak week",
        f"{figures['pre_christmas_lift']:+.0%}",
        help="Best week before Christmas vs a median week (chain-wide sales)",
    )

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
            yaxis={"type": "category", "categoryorder": "total ascending"},
        )
        st.plotly_chart(fig, width="stretch")

    st.write("---")

    st.subheader("Key takeaways")
    st.markdown(
        f"""
- **Seasonality** is the main time pattern: the week before Christmas sells
  {figures['pre_christmas_lift']:+.0%} above a median week, Thanksgiving week
  {figures['thanksgiving_lift']:+.0%}.
- **The dataset's holiday flag is misleading for Christmas**: it marks the week *after*
  Dec 25, which is no busier than a normal week. Plan on the real calendar.
- **Store concentration**: the top 10 stores make {top10_store_share:.0%} of revenue, and
  {len(figures['under_stores'])} stores sit more than 25% below their type's median sales per sq ft.
- **Flat overall growth** ({figures['total_growth']:+.1%}, Feb–Oct 2012 vs 2010), with
  {figures['declining_stores']} stores in decline.
        """
    )

    st.info(
        "Deeper charts → **Visualization**. Data quality & sources → **Exploration**. "
        "Forecasting experiments → **Modeling** (optional).",
        icon="👉",
    )
