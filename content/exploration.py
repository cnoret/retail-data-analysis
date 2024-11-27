"""
Page : Data Exploration
"""

import streamlit as st
import pandas as pd

def exploration():
    "Data Exploration content page"
    st.title("Data Exploration")
    st.info("The data sets are very clean and well structured. Only the features.csv file shows missing values, particularly in the “Markdown” columns, representing weeks when there were simply no promotions running.", icon='✨')

    with st.spinner('Loading Data...⏳'):
        try:
            # Reading .csv files
            df = pd.read_csv('data/merged_retail_data.csv')
            stores = pd.read_csv('data/stores.csv')
            sales = pd.read_csv('data/sales.csv')
            features = pd.read_csv('data/features.csv')

            # Calculating the percentage of missing values in a dataframe
            nan_info = {name: (data.isna().sum(), (data.isna().mean() * 100).round(2)) 
                        for name, data in zip(['Sales', 'Stores', 'Features', 'Merged Data'], [sales, stores, features, df])}

        except pd.errors.EmptyDataError as e:
            st.error(f"An error occurred while reading the CSV file: {str(e)}")
            return

        except FileNotFoundError as e:
            st.error(f"The specified file cannot be found: {str(e)}")
            return

        except KeyError as e:
            st.error(f"The specified column cannot be found in the DataFrame: {str(e)}")
            return

        except Exception as e:
            st.error(f"An unexpected error has occurred: {str(e)}")
            return

    # Display datasets overview with metrics and descriptions
    
    # Stores dataset
    st.write("### Overview of stores.csv")
    st.write("**Description:** Anonymized information about the 45 stores, indicating the type and size of store.")
    st.dataframe(stores.head())
    st.metric(label = "Number of rows", value = stores.shape[0])
    st.metric(label = "Number of columns", value = stores.shape[1])
    st.write("Missing values per column:")
    st.write(stores.isna().sum())
    st.write("---")

    # Features dataset
    st.write("### Overview of features.csv")
    st.write("**Description:** Contains additional data related to the store, department, and regional activity for the given dates.")
    st.write("""
    - **Store:** the store number
    - **Date:** the week
    - **Temperature:** average temperature in the region
    - **Fuel_Price:** cost of fuel in the region
    - **MarkDown1-5:** anonymized data related to promotional markdowns (only available after Nov 2011)
    - **CPI:** the consumer price index
    - **Unemployment:** the unemployment rate
    - **IsHoliday:** whether the week is a special holiday week
    """)
    st.dataframe(features.head())
    st.metric(label = "Number of rows", value = features.shape[0])
    st.metric(label = "Number of columns", value = features.shape[1])
    st.write("Missing values per column:")
    st.write(features.isna().sum())
    st.write("---")

    # Sales dataset
    st.write("### Overview of sales.csv")
    st.write("**Description:** Historical sales data, covering from 2010-02-05 to 2012-11-01.")
    st.write("""
    - **Store:** the store number
    - **Dept:** the department number
    - **Date:** the week
    - **Weekly_Sales:** sales for the given department in the given store
    - **IsHoliday:** whether the week is a special holiday week
    """)
    st.dataframe(sales.head())
    st.metric(label = "Number of rows", value = sales.shape[0])
    st.metric(label = "Number of columns", value = sales.shape[1])
    st.write("Missing values per column:")
    st.write(sales.isna().sum())