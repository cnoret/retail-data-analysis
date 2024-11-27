"""
Page : Data Processing
"""

import streamlit as st
import pandas as pd

def preparation():
    "Data Processing content page"
    
    st.title("Data Processing")
    st.info("Let's clean, transform, and prepare the dataset for analysis !", icon = "🔧")
    
    # Load the datasets
    st.subheader("Loading data")
    
    try:
        features = pd.read_csv('data/features.csv')
        sales = pd.read_csv('data/sales.csv')
        stores = pd.read_csv('data/stores.csv')
        
        st.success("features.csv, sales.csv, stores.csv successfully loaded!", icon = "✅")
        
    except FileNotFoundError as e:
        st.error(f"The specified file cannot be found: {str(e)}")
        return

    st.write("---")
    
    # Missing Values Calculation
    st.subheader("Missing values")

    def display_missing_values(df, name):
        nan_count = df.isna().sum()
        nan_percentage = (df.isna().mean() * 100).round(2)
        nan_info = pd.concat([nan_count, nan_percentage], axis = 1, keys = ['NaN Count', '% of NaN'])
        st.write(f"**{name}.csv :**")
        st.dataframe(nan_info[nan_info['NaN Count'] > 0])
        return nan_info
    
    display_missing_values(features, "features")
    display_missing_values(sales, "sales")
    display_missing_values(stores, "stores")
    
    st.write("---")
    
    # Handling Missing values
    st.subheader("Strategy for missing values")
    st.write("""
    - **MarkDown Columns:** These columns contain anonymized data related to promotional markdowns, which are only available for certain periods and stores. 
      Since missing values in these columns indicate the absence of markdowns, we replace them with 0, which implies no markdown activity.
    - **CPI and Unemployment:** These economic indicators are crucial for understanding the regional activity and consumer behavior. 
      We use forward fill (ffill) to propagate the last known value forward to fill in missing entries, as these values typically change gradually over time.
    """)
    
    # Fill missing values in MarkDown columns with 0
    features[['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']] = features[['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']].fillna(0)
    st.success("Filled missing values in MarkDown columns with 0.", icon = "✅")
    
    # Fill missing values in CPI and Unemployment using ffill
    features[['CPI', 'Unemployment']] = features[['CPI', 'Unemployment']].ffill()
    st.success("Filled missing values in CPI and Unemployment using forward fill.", icon = "✅")
    
    st.write("**Remaining missing values in Features :**")
    st.dataframe(features.isnull().sum()[features.isnull().sum() > 0])
    
    st.write("---")
    
    # Check for duplicates
    st.subheader("Checking duplicates")
    st.write(f"**Number of duplicates in Features:** {features.duplicated().sum()}")
    st.write(f"**Number of duplicates in Sales:** {sales.duplicated().sum()}")
    st.write(f"**Number of duplicates in Stores:** {stores.duplicated().sum()}")
    
    st.write("---")
    
    # Convert Date columns to datetime format
    st.subheader("Convert Date Columns to Datetime Format")
    st.write("Converting the Date columns in Features and Sales datasets to datetime format.")

    try:
        features['Date'] = pd.to_datetime(features['Date'], format='%d/%m/%Y')
        sales['Date'] = pd.to_datetime(sales['Date'], format='%d/%m/%Y')
        st.success("Date columns successfully converted.", icon = "✅")
    except Exception as e:
        st.error(f"An error occurred while converting dates: {str(e)}")
    
    st.write("---")
    
    # Merge Datasets
    st.subheader("Merge datasets")
    st.write("Merging the Sales and Features datasets on Store and Date columns, and then merging the result with the Stores dataset on the Store column.")

    try:
        merged_data = pd.merge(sales, features, on=['Store', 'Date'], how='left')
        merged_data = pd.merge(merged_data, stores, on='Store', how='left')
        st.success("Datasets successfully merged!", icon = "✅")
        
        # Display merged data characteristics
        st.write("**Merged dataset characteristics:**")
        st.dataframe(merged_data.head())

        # Check for missing values in the merged dataset
        st.write("**Missing values in merged dataset :**")
        st.dataframe(merged_data.isnull().sum()[merged_data.isnull().sum() > 0])
        
        # Check for duplicates in the merged dataset
        st.write(f"**Number of duplicates in merged dataset :** {merged_data.duplicated().sum()}")
        
    except Exception as e:
        st.error(f"An error occurred while merging datasets: {str(e)}")
    
    st.write("---")
    
    # Save the merged dataset
    st.subheader("Save the merged dataset")
    st.write("Finally, we save the processed and merged dataset to a new CSV file.")

    try:
        merged_data.to_csv('data/merged_retail_data.csv', index=False)
        st.success("Merged dataset saved successfully as 'merged_retail_data.csv'!", icon = "✅")
    except Exception as e:
        st.error(f"An error occurred while saving the merged dataset: {str(e)}")

    st.write("---")
    
    st.info("Data processing is complete! We can now proceed to the analysis and modeling steps.")

