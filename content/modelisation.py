"""
Page : Modeling and prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

FEATURE_COLS = [
    'Store', 'Dept', 'IsHoliday_x', 'Temperature',
    'Fuel_Price', 'CPI', 'Unemployment', 'Type',
]

FEATURE_LABELS = {
    'Store': 'Store ID',
    'Dept': 'Department',
    'IsHoliday_x': 'Holiday week',
    'Temperature': 'Temperature (°F)',
    'Fuel_Price': 'Fuel Price ($/gal)',
    'CPI': 'Consumer Price Index',
    'Unemployment': 'Unemployment rate',
    'Type': 'Store type (encoded)',
}


@st.cache_data
def load_model_data():
    return pd.read_csv('data/merged_retail_data.csv')


@st.cache_resource
def get_model_pipeline(model_choice):
    data = pd.read_csv('data/merged_retail_data.csv')
    X = data[FEATURE_COLS].copy()
    y = data['Weekly_Sales']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train['Type'] = X_train['Type'].astype('category').cat.codes.astype(int)
    X_test['Type'] = X_test['Type'].astype('category').cat.codes.astype(int)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    if model_choice == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled


def _compute_metrics(y_true, y_pred):
    return {
        "R²": round(r2_score(y_true, y_pred), 4),
        "RMSE ($)": round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        "MAE ($)": round(mean_absolute_error(y_true, y_pred), 2),
    }


def modelisation():
    "Modeling page content"

    st.title("Modeling & Prediction")
    st.caption(
        "Training machine learning models to forecast weekly department-level sales "
        "and evaluating their performance on held-out data."
    )

    try:
        load_model_data()
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return

    # ── 1. Feature selection ──────────────────────────────────────────────────
    st.subheader("1 · Feature Selection")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(
            pd.DataFrame.from_dict(
                FEATURE_LABELS, orient='index', columns=['Description']
            ),
            use_container_width=True,
        )
    with col2:
        st.info(
            "**Excluded features**\n\n"
            "- **Date** — temporal index, seasonal effects captured via `IsHoliday` "
            "and economic indicators.\n"
            "- **MarkDown1–5** — sparse (only available post Nov 2011) and weakly "
            "correlated with sales in our analysis.",
            icon="ℹ️",
        )

    st.write("---")

    # ── 2. Model training ─────────────────────────────────────────────────────
    st.subheader("2 · Model Training")

    model_choice = st.selectbox(
        "Select a model:",
        ("Linear Regression", "Random Forest Regressor"),
    )

    label = (
        f"Training {model_choice}… (~2 min)"
        if model_choice == "Random Forest Regressor"
        else f"Training {model_choice}…"
    )
    with st.spinner(label):
        pipeline = get_model_pipeline(model_choice)
    model, scaler, X_train, X_test, _, y_test, X_train_scaled, X_test_scaled = pipeline

    st.success(f"{model_choice} trained on {len(X_train):,} samples.", icon="✅")

    with st.expander("Preprocessing details"):
        col1, col2 = st.columns(2)
        col1.metric("Training samples", f"{len(X_train):,}")
        col2.metric("Test samples", f"{len(X_test):,}")
        st.caption(
            "80/20 split · `Type` label-encoded · all features standardized (StandardScaler)"
        )
        st.dataframe(
            pd.DataFrame(X_train_scaled[:5], columns=X_train.columns).round(3),
            use_container_width=True,
        )

    st.write("---")

    # ── 3. Evaluation ─────────────────────────────────────────────────────────
    st.subheader("3 · Model Evaluation")

    y_pred = model.predict(X_test_scaled)
    metrics = _compute_metrics(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("R²", f"{metrics['R²']:.4f}")
    col2.metric("RMSE", f"${metrics['RMSE ($)']:,.0f}")
    col3.metric("MAE", f"${metrics['MAE ($)']:,.0f}")

    if model_choice == "Linear Regression" and metrics["R²"] < 0.2:
        st.warning(
            "Linear Regression performs poorly here (R²≈0.06) because `Store` and `Dept` "
            "are numeric IDs — LR treats them as continuous ordered values, which is incorrect. "
            "Random Forest handles this naturally through non-linear splits. "
            "Switch to **Random Forest** to see the difference.",
            icon="⚠️",
        )

    with st.expander("What do these metrics mean?"):
        st.write("""
        - **R²** — proportion of sales variance explained by the model. 1.0 is perfect.
        - **RMSE** — typical prediction error in dollars (penalises large errors more).
        - **MAE** — average absolute error in dollars (more robust to outliers).
        """)

    st.write("")

    # Actual vs Predicted
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_test), size=min(2000, len(y_test)), replace=False)
    scatter_df = pd.DataFrame({
        "Actual ($)": np.array(y_test)[idx],
        "Predicted ($)": y_pred[idx],
    })
    max_val = float(scatter_df.max().max())

    fig = px.scatter(
        scatter_df,
        x="Actual ($)",
        y="Predicted ($)",
        opacity=0.4,
        title="Actual vs Predicted Weekly Sales",
        color_discrete_sequence=["#636EFA"],
    )
    fig.add_shape(
        type="line", x0=0, y0=0, x1=max_val, y1=max_val,
        line=dict(color="red", dash="dash", width=1.5),
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance (RF only)
    if model_choice == "Random Forest Regressor":
        importance_df = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "Importance": model.feature_importances_,
        }).sort_values("Importance")

        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Importance (Random Forest)",
            color="Importance",
            color_continuous_scale="Blues",
        )
        fig.update_layout(coloraxis_showscale=False, height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.write("---")

    # ── 4. Model comparison ───────────────────────────────────────────────────
    st.subheader("4 · Model Comparison")

    rows = []
    for choice in ["Linear Regression", "Random Forest Regressor"]:
        with st.spinner(f"Evaluating {choice}…"):
            m, _, _, _, _, y_t, _, x_t_sc = get_model_pipeline(choice)
        y_p = m.predict(x_t_sc)
        rows.append({"Model": choice, **_compute_metrics(y_t, y_p)})

    comparison = pd.DataFrame(rows).set_index("Model")
    styled = (
        comparison.style
        .highlight_max(axis=0, subset=["R²"], color="#d4edda")
        .highlight_min(axis=0, subset=["RMSE ($)", "MAE ($)"], color="#d4edda")
    )
    st.dataframe(styled, use_container_width=True)

    st.write("---")

    # ── 5. Make a prediction ──────────────────────────────────────────────────
    st.subheader("5 · Make a Prediction")
    st.caption("Use the trained model to predict weekly sales for any store and department.")

    col1, col2 = st.columns(2)
    with col1:
        store = st.number_input("Store (1–45)", min_value=1, max_value=45, value=20)
        dept = st.number_input("Department", min_value=1, max_value=99, value=1)
        is_holiday = st.selectbox(
            "Holiday week?", [False, True], format_func=lambda x: "Yes" if x else "No"
        )
        store_type = st.selectbox("Store type", ["A", "B", "C"])
    with col2:
        temperature = st.number_input("Temperature (°F)", value=60.0)
        fuel_price = st.number_input("Fuel Price ($/gal)", value=3.50)
        cpi = st.number_input("CPI", value=210.0)
        unemployment = st.number_input("Unemployment (%)", value=8.0)

    type_code = {"A": 0, "B": 1, "C": 2}[store_type]

    input_data = pd.DataFrame(
        [[store, dept, int(is_holiday), temperature, fuel_price, cpi, unemployment, type_code]],
        columns=FEATURE_COLS,
    )
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.write("")
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #0071CE 0%, #005BA1 100%);
            border-radius: 14px;
            padding: 32px 24px;
            text-align: center;
            margin-top: 16px;
        ">
            <p style="color:#BFDBFE; font-size:0.85rem; font-weight:600;
                      text-transform:uppercase; letter-spacing:0.08em; margin:0 0 8px 0;">
                Predicted Weekly Sales
            </p>
            <p style="color:#FFFFFF; font-size:3rem; font-weight:800; margin:0; line-height:1.1;">
                ${prediction:,.0f}
            </p>
            <p style="color:#93C5FD; font-size:0.8rem; margin:8px 0 0 0;">
                {model_choice} · Store {store} · Dept {dept}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
