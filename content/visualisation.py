"""
Page : Analysis and visualization
"""

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def visualisation():
    "Analysis and visualization content page"

    st.title("Analysis and visualization")

    # Load the data
    st.subheader("Loading merged_retail_data.csv")
    try:
        merged_data = pd.read_csv('data/merged_retail_data.csv')
        st.success("Data successfully loaded!", icon = "✅")
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return

    st.write("---")
    
    st.subheader("Correlation Matrix")
    
    # Encoding categorical variables
    st.info("Encoding categorical variables before calculating the correlation matrix.", icon = "🔧")
    # Ensure all categorical variables are encoded
    encoded_data = merged_data.copy()
    encoded_data['Type'] = encoded_data['Type'].astype('category').cat.codes
    encoded_data['IsHoliday_x'] = encoded_data['IsHoliday_x'].astype(int)
    encoded_data['IsHoliday_y'] = encoded_data['IsHoliday_y'].astype(int)
    # Convert Date to numeric (timestamp)
    encoded_data['Date'] = pd.to_datetime(encoded_data['Date']).map(pd.Timestamp.timestamp)

    # Correlation matrix
    # Select only numerical data
    numerical_data = encoded_data.select_dtypes(include = ['number'])
    
    cor = numerical_data.corr()
    fig, ax = plt.subplots(figsize = (12, 12))
    sns.heatmap(cor, annot = True, cmap = 'coolwarm', ax = ax)
    plt.title('Correlation Matrix (Encoded Data)', fontsize = 18, pad = 20)
    plt.xticks(rotation = 45, ha = 'right', fontsize = 10)
    plt.yticks(fontsize = 10)
    cbar = ax.collections[0].colorbar
    cbar.set_label('Correlation Coefficient', fontsize = 10, labelpad = 10)
    cbar.ax.tick_params(labelsize = 8)
    sns.despine()
    st.pyplot(fig)

    # Display correlation insights
    st.write("""
    * **Promotions and Sales:** The promotions (MarkDown1 to MarkDown5) have weak positive correlations with Weekly Sales, indicating that while there is some positive impact, it's not very strong.
    * **Temperature and Sales:** Temperature has a very weak negative correlation (-0.02) with Weekly Sales.
    * **Fuel Price and Sales:** Fuel Price shows a weak positive correlation (0.02) with Weekly Sales.
    * **Seasonal Effects:** Date shows moderate to high positive correlations with Temperature (0.14) and Fuel Price (0.47), indicating seasonal effects.
    * **Temperature and Fuel Price:** There is a strong positive correlation (0.30) between Temperature and Fuel Price, which makes sense as fuel prices can be influenced by seasonal demand.
    * **Impact of Promotions:** Promotions have a weak but positive impact on sales, suggesting that while they do help in increasing sales, the effect is not very strong.
    * **Promotions Intercorrelation:** There are stronger intercorrelations among certain promotions, indicating they may be applied together.
    * **Other Factors:** Other factors like temperature, fuel price, and economic indicators (CPI and unemployment) have weak correlations with sales.
    """)
    
    st.write("---")

    # Weekly sales distribution
    st.subheader("Weekly Sales Distribution")
    fig, ax = plt.subplots(figsize = (10, 6))
    plt.hist(merged_data['Weekly_Sales'], bins = 50, edgecolor = 'k', alpha = 0.7)
    plt.title('Weekly Sales Distribution')
    plt.xlabel('Weekly Sales')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle = '--')
    st.pyplot(fig)
    
    st.write("""
    - The distribution of weekly sales is heavily skewed to the right indicating that most sales are concentrated at lower values.
    - There is a high frequency of weeks with relatively low sales, while a few weeks have significantly higher sales, possibly due to promotions or seasonal events.
    - The presence of outliers suggests exceptional weeks with very high sales, which could be driven by factors like holidays or special discounts.
    """)

    st.write("---")

    # Total sales by store
    st.subheader("Total Sales by Store")
    store_sales = merged_data.groupby('Store')['Weekly_Sales'].sum().reset_index()
    sorted_store_sales = store_sales.sort_values(by='Weekly_Sales', ascending = False)

    fig, axes = plt.subplots(2, 1, figsize = (14, 16))
    sns.barplot(ax = axes[0], x = 'Store', y = 'Weekly_Sales', data = store_sales, 
                palette = 'rainbow', hue = 'Store', legend = False)
    axes[0].set_title('Total Sales by Store (Unsorted)', fontsize=18)
    axes[0].set_xlabel('Store')
    axes[0].set_ylabel('Total Sales')
    axes[0].tick_params(axis = 'x')

    sns.barplot(ax = axes[1], x = 'Store', y = 'Weekly_Sales', 
                data = sorted_store_sales, palette = 'rainbow', 
                order = sorted_store_sales['Store'], hue = 'Store', legend = False)
    axes[1].set_title('Total Sales by Store (Descending Order)', fontsize = 18)
    axes[1].set_xlabel('Store')
    axes[1].set_ylabel('Total Sales')
    axes[1].tick_params(axis = 'x')
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig)
    
    st.write("""
    - The top plot shows the total sales for each store without any sorting. We can observe that sales vary significantly across the different stores.
    - The bottom plot, which sorts the stores by total sales in descending order, clearly highlights which stores are the top performers.
    - **Top Performing Stores:** Store 20, Store 4, and Store 14 stand out as the top three performers with the highest total sales. This suggests these stores may be in high-demand locations or have better sales strategies.
    - **Variation Across Stores:** There's a noticeable decline in total sales as we move from the top-performing stores to the lower-performing ones. This indicates a significant disparity in performance across different stores.
    - **Business Implications:** Understanding the factors contributing to the success of the top stores could provide valuable insights for improving sales strategies in underperforming stores.
    """)
    
    st.write("---")

    # Sales trends over time
    st.subheader("Sales Trends Over Time")
    fig, ax = plt.subplots(figsize = (12, 6))
    merged_data.groupby('Date')['Weekly_Sales'].sum().plot(ax = ax)
    plt.title('Sales Trend Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    st.pyplot(fig)
    
    st.write("""
    - The sales trend over time shows noticeable spikes during specific periods, which could correspond to holiday seasons, promotions, or other special events that drive higher sales.
    - The most prominent spikes appear around November 2010, December 2011, and January 2012, likely indicating significant holiday shopping periods such as Thanksgiving, Christmas, and New Year's.
    - Outside of these peak periods, the sales tend to fluctuate within a narrower range, with a slight downward trend observed in certain intervals, possibly due to seasonal effects or economic conditions.
    - **Business Implications:** Understanding the factors driving these sales spikes could help in planning future promotions or stocking strategies to optimize sales during these high-demand periods.
    """)
    
    st.write("---")
    st.info("This page provides visual insights into the data, showcasing trends, distributions, and relationships within the dataset.")

