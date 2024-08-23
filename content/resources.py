"""
Page : Resources
"""

import streamlit as st

def resources():
    "Page content"
    st.title("Resources and Technologies Used")

    st.subheader("Data Sources")
    st.markdown("""
    The data used in this project comes from Kaggle:
    - **[Historical Sales Data from 45 Stores](https://www.kaggle.com/datasets/manjeetsingh/retaildataset/data):** This dataset includes historical sales data along with additional features such as store information and economic indicators, which are crucial for the analysis and forecasting tasks undertaken in this project.
    """)

    st.subheader("Skills and Technologies")

    st.markdown("#### Python 🐍")
    st.markdown("""
    Python was the primary programming language used in this project. Below are the key libraries and frameworks employed:
    
    - **[Pandas](https://pandas.pydata.org/):** Used for data manipulation and analysis, especially for handling structured data in DataFrames.
    - **[NumPy](https://numpy.org/):** Essential for numerical computations and array manipulations.
    - **[Matplotlib](https://matplotlib.org/):** A plotting library used to create static, animated, and interactive visualizations.
    - **[Seaborn](https://seaborn.pydata.org/):** Built on top of Matplotlib, Seaborn is used for making attractive and informative statistical graphics.
    - **[Plotly](https://plotly.com/python/):** Enables the creation of interactive plots and dashboards, providing advanced visualization capabilities.
    - **[Scikit-Learn](https://scikit-learn.org/stable/):** Used for machine learning tasks, including model building, evaluation, and data preprocessing.
    - **[Streamlit](https://streamlit.io/):** A framework for building and deploying interactive web applications directly from Python scripts.
    """)

    st.subheader("Development and Deployment Tools")

    st.markdown("""
    The following tools were integral in the development and deployment of this project:
    
    - **[Jupyter Notebook](https://jupyter.org/):** A web-based interactive computing platform used for writing and executing Python code, performing data analysis, and documenting the process.
    - **[Visual Studio Code](https://code.visualstudio.com/):** A code editor redefined and optimized for building and debugging modern web and cloud applications. It was used extensively for code development.
    - **[Git](https://git-scm.com/):** A version control system that helps in tracking changes to the project code, enabling collaboration, and managing different versions of the project.
    - **[GitHub](https://github.com/):** A platform for hosting and reviewing code, managing projects, and collaborating with others.
    - **[Streamlit Cloud](https://streamlit.io/cloud):** A cloud platform to deploy and share the Streamlit app, making the project accessible to users without needing local setup.
    """)

    st.subheader("Additional Resources")

    st.markdown("""
    - **[Kaggle](https://www.kaggle.com/):** Beyond just providing datasets, Kaggle is a platform for data science competitions and a resource for learning through community and shared notebooks.
    - **[Stack Overflow](https://stackoverflow.com/):** An invaluable resource for troubleshooting coding issues, with a large community contributing solutions to a wide array of programming challenges.
    - **[Jedha](https://www.jedha.co/):** Jedha is a data science bootcamp that provided the foundational knowledge and practical skills applied throughout this project.
    """)

    st.info("These resources were critical in the successful completion of this project, offering tools, knowledge, and community support at every stage.", icon = '✨')