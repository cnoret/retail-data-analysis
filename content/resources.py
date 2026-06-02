"""
Page : Resources
"""

import streamlit as st


def resources():
    "Resources and tech stack page"

    st.title("Stack & Resources")
    st.caption("Libraries, tools, and data sources used in this project.")

    st.write("---")

    # ── Dataset ───────────────────────────────────────────────────────────────
    st.subheader("Dataset")
    st.markdown(
        "**[Walmart Store Sales Forecasting](https://www.kaggle.com/datasets/manjeetsingh/retaildataset/data)** — Kaggle"
    )
    st.write(
        "Anonymized weekly sales data from 45 US retail stores (Feb 2010 – Oct 2012), "
        "including store metadata and economic indicators (CPI, unemployment, fuel price, temperature)."
    )

    st.write("---")

    # ── Tech stack ────────────────────────────────────────────────────────────
    st.subheader("Tech Stack")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Data & ML**")
        st.markdown("""
- [Pandas](https://pandas.pydata.org/) — data manipulation
- [NumPy](https://numpy.org/) — numerical computing
- [Scikit-Learn](https://scikit-learn.org/) — modeling & preprocessing
- [Joblib](https://joblib.readthedocs.io/) — model persistence
        """)

    with col2:
        st.markdown("**Visualization & App**")
        st.markdown("""
- [Plotly](https://plotly.com/python/) — interactive charts
- [Streamlit](https://streamlit.io/) — web app framework
        """)

    st.write("---")

    # ── Tools ─────────────────────────────────────────────────────────────────
    st.subheader("Development & Deployment")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
- [Jupyter Notebook](https://jupyter.org/) — exploratory analysis
- [VS Code](https://code.visualstudio.com/) — development
        """)

    with col2:
        st.markdown("""
- [Git & GitHub](https://github.com/cnoret/retail-data-analysis) — version control
- [Streamlit Cloud](https://streamlit.io/cloud) — deployment
        """)

    st.write("---")
    st.info("Source code available on [GitHub](https://github.com/cnoret/retail-data-analysis).", icon="💻")
