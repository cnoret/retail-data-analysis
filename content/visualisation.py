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
        - **Size** is the only numeric column with a clear link to sales: bigger stores sell more.
          `Store` and `Dept` are **IDs**, so their coefficients mean nothing; what matters is
          *which* store and *which* department (the Store × Dept baseline).
        - **CPI, unemployment, fuel price and temperature** are close to zero
          (|r| < 0.03): the external context does not explain weekly sales.
        - **MarkDown1–5** show a very weak positive correlation (r ≈ 0.02–0.05). Markdowns tend
          to run in busy weeks, so this is not proof of a lift: within each store, high-markdown
          weeks sell the same as other weeks (see `sql/05_markdowns.sql`).
        - **Temperature and Fuel Price** are weakly linked to each other (r ≈ 0.14) and not to sales.
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
        - The high-value outliers come mostly from large (type A) stores in Thanksgiving week
          and the week before Christmas.
        - Modeling keeps the target in dollars and uses a **Store×Dept baseline + residual RF**
          so extreme weeks are explained relative to each department's typical level.
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
        - **Store 20, Store 4, and Store 14** are the top three stores by total revenue, and
          the top 10 make 39% of it.
        - Size drives most of the ranking, so total sales alone is not a fair comparison:
          the Overview compares **sales per square foot** with stores of the same type.
        - On that size-adjusted view, 7 stores sit more than 25% below their type's median.
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
        - Two **annual spikes**: Thanksgiving week (+42–44% vs a median week) and the week
          before Christmas (+67–75%).
        - **January is the low point** of the year, about 10% below a median week, right
          after the peak.
        - The rest of the year is flat, and Feb–Oct 2012 is only +1.9% above Feb–Oct 2010.
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
        - Flagged weeks show a slightly higher median and a wider spread, mostly thanks to
          **Thanksgiving**.
        - The flag misses the real peak: the **week before Christmas** is not flagged, while
          the flagged "Christmas" week (the one ending after Dec 25) sells at or below a
          median week.
        - For planning, use the real calendar (Thanksgiving week and the two weeks before
          Christmas) rather than `IsHoliday`.
        """)
