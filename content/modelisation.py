"""
Page : Modeling
"""

import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    'Store', 'Dept', 'IsHoliday_x', 'Temperature',
    'Fuel_Price', 'CPI', 'Unemployment', 'Type',
    'Week', 'Month',
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
    'Week': 'ISO week of year',
    'Month': 'Month (1–12)',
}

TYPE_CODES = {'A': 0, 'B': 1, 'C': 2}
MODELS_DIR = 'models'
SPLIT_FRAC = 0.8


def _model_key(model_choice):
    return model_choice.lower().replace(' ', '_')


def _model_path(model_choice):
    return os.path.join(MODELS_DIR, f"{_model_key(model_choice)}.joblib")


def _info_path(model_choice):
    return os.path.join(MODELS_DIR, f"{_model_key(model_choice)}_info.json")


def _add_calendar_features(df):
    """Derive Week / Month from Date (in-place safe copy)."""
    out = df.copy()
    dates = pd.to_datetime(out['Date'])
    out['Week'] = dates.dt.isocalendar().week.astype(int)
    out['Month'] = dates.dt.month.astype(int)
    return out


def _encode_type(series):
    """Map store Type A/B/C to stable integer codes."""
    return series.map(TYPE_CODES).astype(int)


@st.cache_data
def load_model_data():
    "Load merged retail dataset."
    return pd.read_csv('data/merged_retail_data.csv')


@st.cache_data
def _prepare_splits():
    """Temporal 80/20 split: train on earlier weeks, test on later weeks."""
    data = _add_calendar_features(pd.read_csv('data/merged_retail_data.csv'))
    data['Date'] = pd.to_datetime(data['Date'])
    data['Type'] = _encode_type(data['Type'])

    unique_dates = np.sort(data['Date'].unique())
    cutoff_idx = int(len(unique_dates) * SPLIT_FRAC)
    cutoff = unique_dates[cutoff_idx]

    train = data[data['Date'] < cutoff]
    test = data[data['Date'] >= cutoff]

    X_train = train[FEATURE_COLS].copy()
    X_test = test[FEATURE_COLS].copy()
    y_train = train['Weekly_Sales'].copy()
    y_test = test['Weekly_Sales'].copy()
    return X_train, X_test, y_train, y_test, pd.Timestamp(cutoff)


def _baseline_predict(X_train, y_train, X_test):
    """Predict mean Weekly_Sales per (Store, Dept) from train; global mean fallback."""
    means = (
        pd.DataFrame({
            'Store': X_train['Store'].values,
            'Dept': X_train['Dept'].values,
            'y': y_train.values,
        })
        .groupby(['Store', 'Dept'])['y']
        .mean()
    )
    global_mean = float(y_train.mean())
    keys = list(zip(X_test['Store'].values, X_test['Dept'].values))
    return np.array([means.get(k, global_mean) for k in keys])


def _train_and_save(model_choice):
    X_train, X_test, y_train, y_test, cutoff = _prepare_splits()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_choice == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(
            n_estimators=20, max_depth=10, n_jobs=1, random_state=42
        )
    model.fit(X_train_scaled, y_train)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(
        {'model': model, 'scaler': scaler, 'feature_cols': FEATURE_COLS},
        _model_path(model_choice),
    )

    y_pred = model.predict(X_test_scaled)
    info = {
        'model': model_choice,
        'trained_at': datetime.now().isoformat(timespec='seconds'),
        'split': 'temporal',
        'cutoff_date': cutoff.strftime('%Y-%m-%d'),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'feature_cols': FEATURE_COLS,
        'r2': round(r2_score(y_test, y_pred), 4),
        'rmse': round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
        'mae': round(mean_absolute_error(y_test, y_pred), 2),
    }
    with open(_info_path(model_choice), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)

    return model, scaler, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, cutoff


def _features_compatible(saved):
    saved_cols = saved.get('feature_cols')
    if saved_cols is None:
        return False
    return list(saved_cols) == FEATURE_COLS


@st.cache_resource
def get_model_pipeline(model_choice):
    "Load model from disk if compatible, otherwise train and save."
    path = _model_path(model_choice)
    if os.path.exists(path):
        saved = joblib.load(path)
        if _features_compatible(saved):
            model, scaler = saved['model'], saved['scaler']
            X_train, X_test, y_train, y_test, cutoff = _prepare_splits()
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            return (
                model, scaler, X_train, X_test, y_train, y_test,
                X_train_scaled, X_test_scaled, cutoff,
            )
    return _train_and_save(model_choice)


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
        "and evaluating their performance on a held-out future period."
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
            width='stretch',
        )
    with col2:
        st.info(
            "**Excluded features**\n\n"
            "- **Date** — replaced by `Week` / `Month` to capture seasonality "
            "without leaking the absolute timestamp.\n"
            "- **MarkDown1–5** — sparse (only available post Nov 2011) and weakly "
            "correlated with sales in our analysis.\n"
            "- **Size** — left out for now; strongly collinear with store identity.",
            icon="ℹ️",
        )

    st.write("---")

    # ── 2. Model training ─────────────────────────────────────────────────────
    st.subheader("2 · Model Training")

    model_choice = st.selectbox(
        "Select a model:",
        ("Linear Regression", "Random Forest Regressor"),
    )
    st.caption(
        "**Linear Regression** is kept as a pedagogical foil "
        "(shows why numeric Store/Dept IDs break a linear model). "
        "**Random Forest** is the serious ML candidate — still judged against the baseline below."
    )

    path = _model_path(model_choice)
    will_load = False
    if os.path.exists(path):
        try:
            will_load = _features_compatible(joblib.load(path))
        except Exception:
            will_load = False

    label = f"Loading {model_choice}…" if will_load else f"Training {model_choice}…"
    with st.spinner(label):
        pipeline = get_model_pipeline(model_choice)
    (
        model, scaler, X_train, X_test, y_train, y_test,
        X_train_scaled, X_test_scaled, cutoff,
    ) = pipeline

    if will_load:
        info_path = _info_path(model_choice)
        if os.path.exists(info_path):
            with open(info_path, encoding='utf-8') as f:
                info = json.load(f)
            st.success(
                f"Model loaded from disk — trained {info['trained_at']} "
                f"on {info['n_train']:,} samples "
                f"(temporal split, cutoff {info.get('cutoff_date', cutoff.date())}).",
                icon="💾",
            )
        else:
            st.success(f"{model_choice} loaded from disk.", icon="💾")
    else:
        st.success(
            f"{model_choice} trained and saved on {len(X_train):,} samples "
            f"(temporal split, cutoff {cutoff.date()}).",
            icon="✅",
        )

    with st.expander("Preprocessing details"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Training samples", f"{len(X_train):,}")
        col2.metric("Test samples", f"{len(X_test):,}")
        col3.metric("Cutoff date", str(cutoff.date()))
        st.caption(
            "Temporal 80/20 split on unique weeks · `Type` mapped A/B/C → 0/1/2 · "
            "`Week`/`Month` from Date · all features standardized (StandardScaler)"
        )
        st.dataframe(
            pd.DataFrame(X_train_scaled[:5], columns=X_train.columns).round(3),
            width='stretch',
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

    if model_choice == "Linear Regression":
        st.warning(
            "**Not a production pick** — Linear Regression is here on purpose as a counter-example. "
            "`Store` and `Dept` are categorical IDs; LR treats them as ordered continuous values, "
            "so it collapses (R²≈0.06). Random Forest splits on them correctly. "
            "Switch to **Random Forest**, then check the comparison table: the real bar is the baseline.",
            icon="⚠️",
        )

    with st.expander("What do these metrics mean?"):
        st.write("""
        - **R²** — proportion of sales variance explained by the model. 1.0 is perfect.
        - **RMSE** — typical prediction error in dollars (penalises large errors more).
        - **MAE** — average absolute error in dollars (more robust to outliers).
        - Metrics are computed on a **future** hold-out (weeks after the cutoff), not a random split.
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
        title="Actual vs Predicted Weekly Sales (temporal hold-out)",
        color_discrete_sequence=["#636EFA"],
    )
    fig.add_shape(
        type="line", x0=0, y0=0, x1=max_val, y1=max_val,
        line={"color": "red", "dash": "dash", "width": 1.5},
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, width='stretch')

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
        st.plotly_chart(fig, width='stretch')

    st.write("---")

    # ── 4. Model comparison ───────────────────────────────────────────────────
    st.subheader("4 · Model Comparison")
    st.info(
        "**How to read this table**\n\n"
        "- **Baseline** — mean weekly sales per (Store, Dept) on the training period only. "
        "This is the bar to beat.\n"
        "- **Linear Regression** — pedagogical foil; expected to fail with raw numeric IDs.\n"
        "- **Random Forest** — non-linear model; useful only if it beats the baseline.",
        icon="ℹ️",
    )

    rows = []
    y_base = _baseline_predict(X_train, y_train, X_test)
    rows.append({"Model": "Baseline (Store × Dept mean)", **_compute_metrics(y_test, y_base)})

    for choice in ["Linear Regression", "Random Forest Regressor"]:
        with st.spinner(f"Evaluating {choice}…"):
            m, _, _, _, _, y_t, _, x_t_sc, _ = get_model_pipeline(choice)
        y_p = m.predict(x_t_sc)
        rows.append({"Model": choice, **_compute_metrics(y_t, y_p)})

    comparison = pd.DataFrame(rows).set_index("Model")
    styled = (
        comparison.style
        .highlight_max(axis=0, subset=["R²"], color="#d4edda")
        .highlight_min(axis=0, subset=["RMSE ($)", "MAE ($)"], color="#d4edda")
    )
    st.dataframe(styled, width='stretch')

    if (
        comparison.loc["Random Forest Regressor", "R²"]
        < comparison.loc["Baseline (Store × Dept mean)", "R²"]
    ):
        st.warning(
            "On this temporal hold-out, **Random Forest does not beat the baseline**. "
            "Sales are largely explained by store/department identity — next gains need "
            "lags or residual modeling on top of the baseline.",
            icon="📌",
        )

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
        week = st.number_input("ISO week", min_value=1, max_value=53, value=26)
    with col2:
        temperature = st.number_input("Temperature (°F)", value=60.0)
        fuel_price = st.number_input("Fuel Price ($/gal)", value=3.50)
        cpi = st.number_input("CPI", value=210.0)
        unemployment = st.number_input("Unemployment (%)", value=8.0)
        month = st.number_input("Month", min_value=1, max_value=12, value=6)

    input_data = pd.DataFrame(
        [[
            store, dept, int(is_holiday), temperature, fuel_price,
            cpi, unemployment, TYPE_CODES[store_type], week, month,
        ]],
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
                {model_choice} · Store {store} · Dept {dept} · W{week}/M{month}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
