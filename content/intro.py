"""
Page : Introduction
"""

import streamlit as st
from PIL import Image


def introduction():
    "Home page content"

    # Displaying the top image with a caption
    image_top = Image.open("images/store.jpg")
    st.image(
        image_top,
        caption="Daniel Orth (www.flickr.com/photos/danorth1/1176509527), Licensed under the Creative Commons Attribution-NoDerivs 2.0 Generic | Flickr",
        width=500,
    )

    # Title and Introduction
    st.subheader("Using Data Science to improve sales strategies")
    st.write(
        """
    **Objective:** To forecast retail sales and analyze the impact of various features on sales.
    
    **Importance:** Accurate sales predictions can optimize inventory management, enhance marketing strategies, and ultimately increase profitability. In this project, we'll use an anonymized dataset collected from 45 stores of a large retail chain to achieve these goals.
    """
    )

    # Analysis Objectives
    st.subheader("Analysis Objectives")
    st.write(
        """
    In this project, we aim to achieve the following key objectives:
    
    1. **Sales Forecasting:** Develop models to predict future sales based on historical data and various influencing factors.
    2. **Feature Impact Analysis:** Evaluate the impact of different features, such as promotions, holidays, and economic indicators, on sales.
    3. **Store Performance Comparison:** Analyze the performance of individual stores and identify factors contributing to higher or lower sales.
    4. **Anomaly Detection:** Detect anomalies in sales patterns, such as unexpected spikes or drops, and explore potential causes.
    """
    )

    # Footer
    st.write("")
    st.info(
        "Let's explore the data and build models to improve sales strategies!",
        icon="🚀",
    )
