"""
Shared data loading for every page.

The merged dataset (~420k rows) is loaded once per server process with
st.cache_resource, so pages share one copy instead of keeping their own.
Pages must treat it as read-only: copy before changing a column.
"""

import pandas as pd
import streamlit as st

MERGED_PATH = "data/merged_retail_data.csv"


@st.cache_resource
def load_merged():
    "Merged sales + features + stores, with Date parsed (the Friday ending each week)."
    return pd.read_csv(MERGED_PATH, parse_dates=["Date"])
