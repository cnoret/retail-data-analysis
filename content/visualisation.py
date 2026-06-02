"""
Page : Visualization
"""

import streamlit as st
import pandas as pd
import plotly.express as px


@st.cache_data
def load_viz_data():
    "Load merged retail dataset."
    return pd.read_csv('data/merged_retail_data.csv')


@st.cache_data
def prepare_correlation_data(df):
    "Encode categorical columns and return numeric correlation matrix."
    encoded = df.copy()
    encoded['Type'] = encoded['Type'].astype('category').cat.codes
    encoded['IsHoliday_x'] = encoded['IsHoliday_x'].astype(int)
    encoded = encoded.drop(columns=['IsHoliday_y'], errors='ignore')
    encoded['Date'] = pd.to_datetime(encoded['Date']).map(pd.Timestamp.timestamp)
    return encoded.select_dtypes(include=['number']).corr()


def visualisation():
    "Visualization content page"

    st.title("Analysis & Visualization")

    try:
        data = load_viz_data()
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return

    # ── 1. Correlation matrix ─────────────────────────────────────────────────
    st.subheader("1 · Correlation Matrix")

    cor = prepare_correlation_data(data)
    fig = px.imshow(
        cor,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Feature Correlation Matrix",
        height=700,
    )
    fig.update_layout(margin={"l": 0, "r": 0})
    st.plotly_chart(fig, width='stretch')

    with st.expander("Key insights"):
        st.write("""
        - **Store & Dept** show the strongest correlation with `Weekly_Sales`, confirming that
          store size and department type are the primary sales drivers.
        - **MarkDown1–5** have weak positive correlations with sales — promotions help,
          but their effect is modest and inconsistent.
        - **CPI and Unemployment** are negatively correlated with sales, reflecting the impact
          of broader economic conditions.
        - **Temperature and Fuel Price** are moderately correlated with each other (seasonal co-movement),
          but both have near-zero direct impact on sales.
        """)

    st.write("---")

    # ── 2. Weekly sales distribution ─────────────────────────────────────────
    st.subheader("2 · Weekly Sales Distribution")

    fig = px.histogram(
        data,
        x='Weekly_Sales',
        nbins=60,
        title="Distribution of Weekly Sales",
        labels={'Weekly_Sales': 'Weekly Sales ($)', 'count': 'Frequency'},
        color_discrete_sequence=['#636EFA'],
    )
    fig.update_layout(bargap=0.05)
    st.plotly_chart(fig, width='stretch')

    with st.expander("Key insights"):
        st.write("""
        - The distribution is **heavily right-skewed**: the vast majority of weekly sales
          fall below $100K, while a long tail extends to over $500K.
        - The high-value outliers correspond to large-format stores (Type A) during
          peak holiday weeks.
        - This skew suggests that a **log transformation** could improve model performance
          by reducing the influence of extreme values.
        """)

    st.write("---")

    # ── 3. Total sales by store ───────────────────────────────────────────────
    st.subheader("3 · Total Sales by Store")

    store_sales = (
        data.groupby('Store')['Weekly_Sales']
        .sum()
        .reset_index()
        .sort_values('Weekly_Sales', ascending=False)
    )

    fig = px.bar(
        store_sales,
        x='Store',
        y='Weekly_Sales',
        title="Total Sales by Store (Descending)",
        labels={'Weekly_Sales': 'Total Sales ($)', 'Store': 'Store'},
        color='Weekly_Sales',
        color_continuous_scale='Blues',
    )
    fig.update_layout(coloraxis_showscale=False, xaxis={'categoryorder': 'total descending'})
    st.plotly_chart(fig, width='stretch')

    with st.expander("Key insights"):
        st.write("""
        - **Store 20, Store 4, and Store 14** are the top three performers by total revenue.
        - There is a sharp drop-off after the top ~10 stores, suggesting a two-tier performance gap.
        - Store type (A / B / C) and surface area are likely the main differentiators —
          larger stores naturally capture more foot traffic.
        """)

    st.write("---")

    # ── 4. Sales trend over time ──────────────────────────────────────────────
    st.subheader("4 · Sales Trend Over Time")

    time_data = (
        data.groupby('Date')['Weekly_Sales']
        .sum()
        .reset_index()
    )
    time_data['Date'] = pd.to_datetime(time_data['Date'])

    fig = px.line(
        time_data,
        x='Date',
        y='Weekly_Sales',
        title="Total Weekly Sales — 2010 to 2012",
        labels={'Weekly_Sales': 'Total Sales ($)', 'Date': ''},
    )
    fig.update_traces(line_color='#636EFA')
    st.plotly_chart(fig, width='stretch')

    with st.expander("Key insights"):
        st.write("""
        - Clear **annual spikes** around November–December each year, driven by
          Thanksgiving and Christmas shopping.
        - A secondary spike is visible in January, likely driven by post-holiday
          clearance sales and New Year promotions.
        - Outside peak periods, sales remain relatively stable with low volatility,
          suggesting a predictable baseline that models can learn effectively.
        """)

    st.write("---")

    # ── 5. Holiday impact ─────────────────────────────────────────────────────
    st.subheader("5 · Holiday vs Non-Holiday Sales")

    fig = px.box(
        data,
        x='IsHoliday_x',
        y='Weekly_Sales',
        color='IsHoliday_x',
        title="Weekly Sales Distribution by Holiday Week",
        labels={'IsHoliday_x': 'Holiday Week', 'Weekly_Sales': 'Weekly Sales ($)'},
        category_orders={'IsHoliday_x': [False, True]},
        color_discrete_map={False: '#636EFA', True: '#EF553B'},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width='stretch')

    with st.expander("Key insights"):
        st.write("""
        - Holiday weeks show a **higher median and a wider spread** than non-holiday weeks,
          confirming that holidays meaningfully boost sales.
        - The effect is most pronounced in the upper tail — some holiday weeks produce
          exceptional volumes not seen in regular weeks.
        - This makes `IsHoliday` a valuable feature for the prediction model.
        """)
