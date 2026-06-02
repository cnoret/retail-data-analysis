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

    st.title("Walmart Sales Forecasting")
    st.caption("Predicting weekly retail sales across 45 stores using machine learning")

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
        This project analyzes **3 years of anonymized sales data** (Feb 2010 – Oct 2012)
        from a major US retail chain. The goal is to build models that accurately forecast
        weekly department-level sales, enabling better inventory planning and marketing decisions.
        """
    )

    st.subheader("Objectives")
    st.write(
        """
        1. **Sales Forecasting** — Predict weekly sales per store and department using historical data and external factors.
        2. **Feature Impact Analysis** — Quantify the effect of promotions, holidays, and economic indicators on sales.
        3. **Store Performance Comparison** — Identify top-performing stores and the drivers behind their results.
        4. **Anomaly Detection** — Surface unexpected sales spikes or drops and explore their causes.
        """
    )

    st.write("")
    st.info(
        "Navigate through the sections using the sidebar to explore the full analysis.",
        icon="👈",
    )
