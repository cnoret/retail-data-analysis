"""
app.py : Main
"""

## Streamlit & UI
import streamlit as st
from streamlit_option_menu import option_menu

## Importing pages
from content.intro import introduction
from content.exploration import exploration
from content.preparation import preparation
from content.visualisation import visualisation
from content.modelisation import modelisation
from content.resources import resources

## Page title & favicon
st.set_page_config(page_title="Walmart Sales Forecasting", page_icon="images/favicon.png")

## Global CSS
st.markdown("""
<style>
/* KPI metric cards */
[data-testid="stMetric"] {
    background-color: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 10px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] {
    font-size: 0.82rem;
    color: #64748B;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stMetricValue"] {
    font-size: 1.8rem;
    color: #0071CE;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

## Sidebar menu
with st.sidebar:
    st.image("images/trolley.png", width=72)
    st.header("Walmart Sales Forecasting")
    choice = option_menu(
        menu_title="Summary",
        options=[
            "Introduction",
            "Data Exploration",
            "Data Processing",
            "Visualization",
            "Modeling",
            "Resources",
        ],
        icons=[
            "house",
            "search",
            "gear",
            "bar-chart",
            "cpu",
            "book",
        ],
        default_index=0,
    )

    st.write("---")
    st.markdown(
        "<p style='text-align:center; font-size:0.85rem; color:gray; margin-bottom:6px;'>"
        "Christophe NORET</p>",
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    col1.link_button("LinkedIn", "https://www.linkedin.com/in/christophenoret", width='stretch')
    col2.link_button("GitHub", "https://github.com/cnoret", width='stretch')

## Main Menu
if choice == "Introduction":
    introduction()

elif choice == "Data Exploration":
    exploration()

elif choice == "Data Processing":
    preparation()

elif choice == "Visualization":
    visualisation()

elif choice == "Modeling":
    modelisation()

else:
    resources()
