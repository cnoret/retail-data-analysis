"""
Page : Data Exploration
"""

import streamlit as st
import pandas as pd


@st.cache_data
def load_exploration_data():
    df = pd.read_csv('data/merged_retail_data.csv')
    stores = pd.read_csv('data/stores.csv')
    sales = pd.read_csv('data/sales.csv')
    features = pd.read_csv('data/features.csv')
    return df, stores, sales, features


def _dataset_section(name, df, description, column_descriptions=None):
    st.subheader(f"{name}")
    st.caption(description)

    col1, col2, col3 = st.columns(3)
    missing_pct = (df.isna().mean() * 100).max()
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", df.shape[1])
    col3.metric("Max missing", f"{missing_pct:.1f}%")

    if column_descriptions:
        with st.expander("Column descriptions"):
            for col, desc in column_descriptions.items():
                st.markdown(f"- **{col}:** {desc}")

    st.dataframe(df.head(), use_container_width=True)

    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        st.success("No missing values.", icon="✅")
    else:
        with st.expander(f"Missing values ({len(missing)} column(s))"):
            pct = (df.isna().mean() * 100).round(2)
            summary = pd.DataFrame({"Count": missing, "% of rows": pct[missing.index]})
            st.dataframe(summary)

    st.write("---")


def exploration():
    "Data Exploration content page"

    st.title("Data Exploration")
    st.info(
        'The datasets are clean and well-structured. Only **features.csv** contains missing values, '
        'in the "MarkDown" columns — weeks with no active promotions.',
        icon='✨',
    )

    try:
        df, stores, sales, features = load_exploration_data()
    except pd.errors.EmptyDataError as e:
        st.error(f"An error occurred while reading the CSV file: {str(e)}")
        return
    except FileNotFoundError as e:
        st.error(f"The specified file cannot be found: {str(e)}")
        return
    except Exception as e:
        st.error(f"An unexpected error has occurred: {str(e)}")
        return

    _dataset_section(
        "stores.csv",
        stores,
        "Anonymized information about the 45 stores: type (A / B / C) and surface area.",
    )

    _dataset_section(
        "features.csv",
        features,
        "Weekly contextual data per store: economic indicators, fuel price, temperature, and promotional markdowns.",
        column_descriptions={
            "Store": "Store number",
            "Date": "Week start date",
            "Temperature": "Average regional temperature (°F)",
            "Fuel_Price": "Regional fuel price ($/gallon)",
            "MarkDown1-5": "Anonymized promotional markdown data (available from Nov 2011 only)",
            "CPI": "Consumer Price Index",
            "Unemployment": "Regional unemployment rate",
            "IsHoliday": "Whether the week includes a major holiday",
        },
    )

    _dataset_section(
        "sales.csv",
        sales,
        "Historical weekly sales per store and department, covering Feb 2010 – Oct 2012.",
        column_descriptions={
            "Store": "Store number",
            "Dept": "Department number",
            "Date": "Week start date",
            "Weekly_Sales": "Sales for the given department and store (USD)",
            "IsHoliday": "Whether the week includes a major holiday",
        },
    )

    # Merged dataset summary
    st.subheader("Merged dataset")
    st.caption("Result of joining sales, features, and stores on Store and Date.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total rows", f"{len(df):,}")
    col2.metric("Stores", df["Store"].nunique())
    col3.metric("Departments", df["Dept"].nunique())
    col4.metric("Weeks covered", df["Date"].nunique())

    st.dataframe(df.head(), use_container_width=True)
    st.info(
        "This merged dataset is the single source used for all subsequent analysis and modeling steps.",
        icon="ℹ️",
    )
