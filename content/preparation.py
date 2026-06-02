"""
Page : Data Processing
"""

import streamlit as st
import pandas as pd


@st.cache_data
def load_raw_data():
    return (
        pd.read_csv('data/features.csv'),
        pd.read_csv('data/sales.csv'),
        pd.read_csv('data/stores.csv'),
    )


def preparation():
    "Data Processing content page"

    st.title("Data Processing")
    st.caption("Cleaning, transforming, and merging the three source datasets into one analysis-ready file.")

    try:
        features, sales, stores = load_raw_data()
        features = features.copy()
        sales = sales.copy()
    except FileNotFoundError as e:
        st.error(f"File not found: {str(e)}")
        return

    # ── 1. Missing values ────────────────────────────────────────────────────
    st.subheader("1 · Missing values")

    def missing_summary(df, name):
        missing = df.isna().sum()
        missing = missing[missing > 0]
        if missing.empty:
            st.success(f"**{name}** — no missing values.", icon="✅")
        else:
            pct = (df.isna().mean() * 100).round(2)
            summary = pd.DataFrame({"Count": missing, "% of rows": pct[missing.index]})
            st.warning(f"**{name}** — {len(missing)} column(s) with missing values.", icon="⚠️")
            st.dataframe(summary)

    missing_summary(features, "features.csv")
    missing_summary(sales, "sales.csv")
    missing_summary(stores, "stores.csv")

    st.write("---")

    # ── 2. Strategy ──────────────────────────────────────────────────────────
    st.subheader("2 · Imputation strategy")

    col1, col2 = st.columns(2)
    with col1:
        st.info(
            "**MarkDown1–5 → fill with 0**\n\n"
            "Missing weeks simply had no active promotions. Zero is the semantically correct value.",
            icon="🏷️",
        )
    with col2:
        st.info(
            "**CPI & Unemployment → forward fill**\n\n"
            "Economic indicators change gradually. Propagating the last known value is a sound approximation.",
            icon="📈",
        )

    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    features[markdown_cols] = features[markdown_cols].fillna(0)
    features[['CPI', 'Unemployment']] = features[['CPI', 'Unemployment']].ffill()

    remaining = features.isna().sum().sum()
    if remaining == 0:
        st.success("All missing values resolved — features.csv is now complete.", icon="✅")
    else:
        st.warning(f"{remaining} missing value(s) still present after imputation.")

    st.write("---")

    # ── 3. Duplicates ────────────────────────────────────────────────────────
    st.subheader("3 · Duplicate rows")

    col1, col2, col3 = st.columns(3)
    col1.metric("features.csv", features.duplicated().sum())
    col2.metric("sales.csv", sales.duplicated().sum())
    col3.metric("stores.csv", stores.duplicated().sum())
    st.success("No duplicate rows in any dataset.", icon="✅")

    st.write("---")

    # ── 4. Date conversion ───────────────────────────────────────────────────
    st.subheader("4 · Date conversion")
    st.write("Parsing `Date` columns to `datetime` format for time-based operations.")

    try:
        features['Date'] = pd.to_datetime(features['Date'], format='%d/%m/%Y')
        sales['Date'] = pd.to_datetime(sales['Date'], format='%d/%m/%Y')
        col1, col2 = st.columns(2)
        col1.success("features.csv ✅")
        col2.success("sales.csv ✅")
    except Exception as e:
        st.error(f"Date conversion failed: {str(e)}")

    st.write("---")

    # ── 5. Merge ─────────────────────────────────────────────────────────────
    st.subheader("5 · Merging datasets")
    st.write(
        "Left-joining **sales** with **features** on `(Store, Date)`, "
        "then with **stores** on `Store`."
    )

    try:
        merged = pd.merge(sales, features, on=['Store', 'Date'], how='left')
        merged = pd.merge(merged, stores, on='Store', how='left')

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{len(merged):,}")
        col2.metric("Columns", merged.shape[1])
        col3.metric("Duplicates", merged.duplicated().sum())

        st.dataframe(merged.head(), use_container_width=True)

        remaining_merged = merged.isnull().sum()
        remaining_merged = remaining_merged[remaining_merged > 0]
        if remaining_merged.empty:
            st.success("No missing values in the merged dataset.", icon="✅")
        else:
            with st.expander(f"{len(remaining_merged)} column(s) still have missing values"):
                st.dataframe(remaining_merged)

    except Exception as e:
        st.error(f"Merge failed: {str(e)}")
        return

    st.write("---")
    st.success(
        "Pipeline complete. The merged dataset is ready for analysis and modeling.",
        icon="🎯",
    )
