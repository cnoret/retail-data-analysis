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
    col1.link_button("LinkedIn", "https://www.linkedin.com/in/christophenoret", use_container_width=True)
    col2.link_button("GitHub", "https://github.com/cnoret", use_container_width=True)

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
