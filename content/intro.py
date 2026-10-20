"""
Page : Introduction
"""

import streamlit as st


def introduction():
    "Home page content"

    st.image(
        "images/store.jpg",
        width=700,
        caption="Daniel Orth (www.flickr.com/photos/danorth1/1176509527), CC BY-ND 2.0 | Flickr",
    )

    st.title("Walmart Retail Sales Analysis")
    st.caption(
        "Explore weekly sales across 45 stores: KPIs, seasonality, store performance, and data quality"
    )

    st.write("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stores", "45")
    col2.metric("Departments", "81")
    col3.metric("Weekly records", "421 K")
    col4.metric("Total revenue", "$6.7 B")

    st.write("---")

    st.subheader("Project overview")
    st.write(
        """
        This project analyzes **3 years of anonymized retail sales** (Feb 2010 – Oct 2012)
        from a major US chain. The focus is **decision-oriented analysis**: understand drivers
        of weekly department sales, compare stores, and surface seasonality and holiday effects
        in an interactive app.
        """
    )

    st.subheader("Objectives")
    st.write(
        """
        1. **Describe the business**: volume, store mix, and department coverage.
        2. **Explain variation**: holidays, seasonality, and store rankings.
        3. **Deliver insights in a dashboard** that a non-technical stakeholder can read in one screen.
        """
    )

    st.write("")
    st.info(
        "Start with **Dashboard** for the executive summary, then Exploration and Visualization. "
        "**Modeling** is an optional annex (does a forecast beat a simple Store×Dept baseline?).",
        icon="👈",
    )
