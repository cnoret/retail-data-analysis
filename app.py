"""
app.py : Main
"""

## Streamlit & UI
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

## Importing pages
from content.intro import introduction
from content.exploration import exploration
from content.preparation import preparation
from content.visualisation import visualisation
from content.modelisation import modelisation
from content.resources import resources

## Page title & favicon
st.set_page_config(page_title = "Retail Sales Analysis", page_icon = "images/favicon.png")

## Sidebar menu
with st.sidebar:
    image_side = Image.open("images/trolley.png")
    st.image(image_side)
    st.header("Walmart Sales Prediction")
    choice = option_menu(
        menu_title = "Summary",
        options = ["Introduction",
                   "Data Exploration",
                   "Data Processing",
                   "Analysis and visualization",
                   "Modeling and prediction",
                   "Resources"],
        default_index = 0)

    # Author
    st.header("Author :")
    st.markdown('Christophe NORET&nbsp;&nbsp;[<img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg" width=25>](http://www.linkedin.com/in/christophenoret) [<img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width=25>](https://github.com/cnoret)', unsafe_allow_html=True)

## Main Menu
if choice == "Introduction":
    introduction()

elif choice == "Data Exploration":
    exploration()

elif choice == "Data Processing":
    preparation()

elif choice == "Analysis and visualization":
    visualisation()

elif choice == "Modeling and prediction":
    modelisation()

else:
    resources()
