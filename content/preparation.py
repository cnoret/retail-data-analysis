"""
Page : Data Processing
"""

import streamlit as st
import pandas as pd


@st.cache_data
def load_raw_data():
    "Load raw source CSV files."
    return (
        pd.read_csv('data/features.csv'),
        pd.read_csv('data/sales.csv'),
        pd.read_csv('data/stores.csv'),
    )


def preparation():
    "Data Processing content page"

    st.title("Data Processing")
    st.caption(
        "Walkthrough of the cleaning and merge steps that produced "
        "`data/merged_retail_data.csv` (the file used by Visualization and Modeling)."
    )
    st.info(
        "This page **recomputes the pipeline in-session for illustration**. "
        "It does not overwrite the committed CSV. Downstream pages always read "
        "`merged_retail_data.csv`.",
        icon="ℹ️",
    )

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
            st.success(f"**{name}**: no missing values.", icon="✅")
        else:
            pct = (df.isna().mean() * 100).round(2)
            summary = pd.DataFrame({"Count": missing, "% of rows": pct[missing.index]})
            st.warning(f"**{name}**: {len(missing)} column(s) with missing values.", icon="⚠️")
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
            "**CPI & Unemployment → forward fill per store**\n\n"
            "Economic indicators change gradually. We sort by date and propagate "
            "the last known value **within each store** (avoids cross-store leakage).",
            icon="📈",
        )

    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    features[markdown_cols] = features[markdown_cols].fillna(0)
    st.success("MarkDown columns filled with 0.", icon="✅")
    st.caption("CPI / Unemployment forward-fill runs after date parsing (step 4).")

    st.write("---")

    # ── 3. Duplicates ────────────────────────────────────────────────────────
    st.subheader("3 · Duplicate rows")

    dup_features = int(features.duplicated().sum())
    dup_sales = int(sales.duplicated().sum())
    dup_stores = int(stores.duplicated().sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("features.csv", dup_features)
    col2.metric("sales.csv", dup_sales)
    col3.metric("stores.csv", dup_stores)

    total_dups = dup_features + dup_sales + dup_stores
    if total_dups == 0:
        st.success("No duplicate rows in any dataset.", icon="✅")
    else:
        st.warning(f"{total_dups} duplicate row(s) detected across source files.", icon="⚠️")

    st.write("---")

    # ── 4. Date conversion + per-store ffill ─────────────────────────────────
    st.subheader("4 · Date conversion & economic forward-fill")
    st.write(
        "Parsing `Date` columns to `datetime`, then forward-filling "
        "`CPI` and `Unemployment` within each store."
    )

    try:
        features['Date'] = pd.to_datetime(features['Date'], format='%d/%m/%Y')
        sales['Date'] = pd.to_datetime(sales['Date'], format='%d/%m/%Y')
        col1, col2 = st.columns(2)
        col1.success("features.csv dates parsed")
        col2.success("sales.csv dates parsed")

        features = features.sort_values(['Store', 'Date'])
        features[['CPI', 'Unemployment']] = (
            features.groupby('Store', group_keys=False)[['CPI', 'Unemployment']].ffill()
        )
        # trailing gaps within a store: back-fill after forward-fill
        features[['CPI', 'Unemployment']] = (
            features.groupby('Store', group_keys=False)[['CPI', 'Unemployment']].bfill()
        )

        remaining = int(features[['CPI', 'Unemployment']].isna().sum().sum())
        if remaining == 0:
            st.success(
                "CPI and Unemployment complete after per-store forward/back fill.",
                icon="✅",
            )
        else:
            st.warning(f"{remaining} CPI/Unemployment value(s) still missing after fill.")
    except Exception as e:
        st.error(f"Date conversion or forward-fill failed: {str(e)}")
        return

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
        col3.metric("Duplicates", int(merged.duplicated().sum()))

        st.dataframe(merged.head(), width='stretch')

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
        "Pipeline demo complete. Visualization and Modeling continue to use "
        "the committed file `data/merged_retail_data.csv`.",
        icon="🎯",
    )
