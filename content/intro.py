"""
Page : Overview (the brief and the recommendations)
"""

import pandas as pd
import streamlit as st

UNDERPERFORMANCE_THRESHOLD = 0.75  # below 75% of the type median sales per sq ft
SHRINKING_DEPT_THRESHOLD = -0.20  # departments that lost more than 20% (Feb-Oct 2012 vs 2010)
SHRINKING_DEPT_MIN_LOSS = 1e6  # ... and more than $1M, to leave out tiny departments


@st.cache_data
def _load_overview_data():
    df = pd.read_csv("data/merged_retail_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data
def compute_headline_figures(df):
    "Compute the figures quoted in the recommendations."
    n_weeks = df["Date"].nunique()
    years = n_weeks / 52

    # Store productivity: sales per sq ft compared with the median store of the same type
    stores = df.groupby(["Store", "Type", "Size"], as_index=False)["Weekly_Sales"].sum()
    stores["sales_per_sqft"] = stores["Weekly_Sales"] / stores["Size"]
    stores["type_median"] = stores.groupby("Type")["sales_per_sqft"].transform("median")
    under = stores[stores["sales_per_sqft"] < UNDERPERFORMANCE_THRESHOLD * stores["type_median"]]
    upside_per_year = (
        (under["type_median"] - under["sales_per_sqft"]) * under["Size"]
    ).sum() / years

    # Year-over-year on comparable months (Feb-Oct, the months covered every year)
    comparable = df[df["Date"].dt.month.between(2, 10)]
    by_store_year = comparable.pivot_table(
        index="Store", columns=comparable["Date"].dt.year, values="Weekly_Sales", aggfunc="sum"
    )
    growth = by_store_year[2012] / by_store_year[2010] - 1
    total_growth = by_store_year[2012].sum() / by_store_year[2010].sum() - 1

    # Holiday peak: weekly chain revenue vs a median week
    weekly = df.groupby("Date")["Weekly_Sales"].sum()
    median_week = weekly.median()
    pre_christmas = weekly[(weekly.index.month == 12) & (weekly.index.day >= 18) & (weekly.index.day <= 24)]
    thanksgiving = weekly[(weekly.index.month == 11) & (weekly.index.day >= 22)]

    # Store ranking by total revenue (1 = largest)
    store_rank = (
        df.groupby("Store")["Weekly_Sales"].sum().rank(ascending=False, method="min").astype(int)
    )

    # Department mix: share of revenue and change on comparable months
    dept_share = df.groupby("Dept")["Weekly_Sales"].sum().sort_values(ascending=False)
    by_dept_year = comparable.pivot_table(
        index="Dept", columns=comparable["Date"].dt.year, values="Weekly_Sales", aggfunc="sum"
    ).fillna(0)
    dept_change = by_dept_year[2012] - by_dept_year[2010]
    dept_growth = by_dept_year[2012] / by_dept_year[2010].where(by_dept_year[2010] > 0) - 1
    growth_depts = dept_change.nlargest(3)
    shrinking = dept_growth[
        (dept_growth < SHRINKING_DEPT_THRESHOLD) & (dept_change < -SHRINKING_DEPT_MIN_LOSS)
    ]
    shrinking = shrinking.loc[dept_change[shrinking.index].sort_values().index]

    return {
        "annual_revenue": df["Weekly_Sales"].sum() / years,
        "under_stores": sorted(under["Store"].tolist()),
        "upside_per_year": upside_per_year,
        "declining_stores": int((growth < 0).sum()),
        "store_14_growth": growth.get(14, float("nan")),
        "total_growth": total_growth,
        "pre_christmas_lift": pre_christmas.max() / median_week - 1,
        "thanksgiving_lift": thanksgiving.max() / median_week - 1,
        "store_14_rank": int(store_rank.get(14, 0)),
        "top10_dept_share": dept_share.head(10).sum() / dept_share.sum(),
        "growth_depts": [int(d) for d in growth_depts.index],
        "growth_depts_change": float(growth_depts.sum()),
        "shrinking_depts": [int(d) for d in shrinking.index],
        "shrinking_depts_growth": [float(g) for g in shrinking],
    }


def introduction():
    "Home page: the brief, headline figures and recommendations"

    st.image(
        "images/store.jpg",
        width=700,
        caption="Daniel Orth (www.flickr.com/photos/danorth1/1176509527), CC BY-ND 2.0 | Flickr",
    )

    st.title("Walmart Store Performance Review")
    st.caption("45 stores · 81 departments · Feb 2010 – Oct 2012 · weekly sales (weeks end on Friday)")

    try:
        data = _load_overview_data()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    figures = compute_headline_figures(data)

    st.subheader("The brief")
    st.write(
        """
        A regional manager overseeing **45 stores** needs to set priorities for next year:
        which stores underperform, which departments to back, how to prepare for the
        holiday season, and whether markdown promotions are worth their cost.
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Revenue / year", f"${figures['annual_revenue'] / 1e9:.2f}B")
    col2.metric("Growth", f"{figures['total_growth']:+.1%}", help="Feb–Oct 2012 vs Feb–Oct 2010")
    col3.metric("Stores down", f"{figures['declining_stores']} / 45")
    col4.metric("Upside / year", f"${figures['upside_per_year'] / 1e6:.0f}M", help="Bringing the underperforming stores to their type median")

    st.write("---")

    under_list = ", ".join(str(s) for s in figures["under_stores"])
    growth_list = ", ".join(str(d) for d in figures["growth_depts"])
    shrinking_list = " and ".join(
        f"{d} ({g:+.0%})"
        for d, g in zip(figures["shrinking_depts"], figures["shrinking_depts_growth"])
    )
    st.subheader("Recommendations")
    st.markdown(
        f"""
1. **Audit {len(figures['under_stores'])} underperforming stores first.**
   Stores {under_list} sell more than 25% less per square foot than the median store of
   their type. Bringing them to that median is worth up to
   **\\${figures['upside_per_year'] / 1e6:.0f}M a year**.
2. **Investigate the decline of large stores.** {figures['declining_stores']} of 45 stores sold
   less in Feb–Oct 2012 than in 2010. Store 14, the #{figures['store_14_rank']} store by revenue,
   changed by **{figures['store_14_growth']:+.0%}**.
3. **Plan the holiday peak on the real calendar.** The week before Christmas sells
   **{figures['pre_christmas_lift']:+.0%}** above a median week and Thanksgiving week
   **{figures['thanksgiving_lift']:+.0%}**. The dataset's "Christmas" flag marks the week
   *after* Dec 25, which is no busier than a normal week.
4. **Back the departments that carry growth.** The top 10 departments make
   **{figures['top10_dept_share']:.0%} of revenue**. Departments {growth_list} drove most of
   the growth (**+\\${figures['growth_depts_change'] / 1e6:.0f}M**), while departments
   {shrinking_list} shrank.
5. **Test markdowns before scaling them.** Markdown weeks show no measurable lift outside
   holidays: a controlled store test is needed before spending more.
        """
    )

    st.info(
        "The **Dashboard** gives the one-screen view. **Visualization** holds the evidence "
        "behind each recommendation, and **Data Exploration** / **Data Processing** document "
        "data quality. **Modeling** is a short appendix.",
        icon="👈",
    )
