"""
Page : Modeling
"""

import json
import os
from datetime import date, datetime

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
MODEL_CHOICES = (
    "Linear Regression",
    "Random Forest Regressor",
    "Residual Random Forest",
)
RF_PARAMS = dict(n_estimators=20, max_depth=10, n_jobs=1, random_state=42)


def _model_key(model_choice):
    return model_choice.lower().replace(' ', '_')


def _model_path(model_choice):
    return os.path.join(MODELS_DIR, f"{_model_key(model_choice)}.joblib")


def _info_path(model_choice):
    return os.path.join(MODELS_DIR, f"{_model_key(model_choice)}_info.json")


def _add_calendar_features(df):
    """Derive Week / Month from Date."""
    out = df.copy()
    dates = pd.to_datetime(out['Date'])
    out['Week'] = dates.dt.isocalendar().week.astype(int)
    out['Month'] = dates.dt.month.astype(int)
    return out


def _encode_type(series):
    return series.map(TYPE_CODES).astype(int)


def _means_to_dict(means):
    return {f"{int(s)}_{int(d)}": float(v) for (s, d), v in means.items()}


def _baseline_from_means(means_dict, global_mean, store, dept):
    return means_dict.get(f"{int(store)}_{int(dept)}", global_mean)


def _baseline_vector(means_dict, global_mean, stores, depts):
    return np.array([
        _baseline_from_means(means_dict, global_mean, s, d)
        for s, d in zip(stores, depts)
    ])


@st.cache_data
def load_model_data():
    "Load merged retail dataset."
    return pd.read_csv('data/merged_retail_data.csv')


@st.cache_data
def _prepare_splits():
    """Temporal 80/20 split; drop negative Weekly_Sales (returns/adjustments)."""
    data = _add_calendar_features(pd.read_csv('data/merged_retail_data.csv'))
    data['Date'] = pd.to_datetime(data['Date'])
    data['Type'] = _encode_type(data['Type'])
    data['IsHoliday_x'] = data['IsHoliday_x'].astype(int)
    n_neg = int((data['Weekly_Sales'] < 0).sum())
    data = data[data['Weekly_Sales'] >= 0].copy()

    unique_dates = np.sort(data['Date'].unique())
    cutoff_idx = int(len(unique_dates) * SPLIT_FRAC)
    cutoff = unique_dates[cutoff_idx]

    train = data[data['Date'] < cutoff]
    test = data[data['Date'] >= cutoff]

    X_train = train[FEATURE_COLS].copy()
    X_test = test[FEATURE_COLS].copy()
    y_train = train['Weekly_Sales'].copy()
    y_test = test['Weekly_Sales'].copy()
    return X_train, X_test, y_train, y_test, pd.Timestamp(cutoff), n_neg


def _fit_baseline_means(X_train, y_train):
    means = (
        pd.DataFrame({
            'Store': X_train['Store'].values,
            'Dept': X_train['Dept'].values,
            'y': y_train.values,
        })
        .groupby(['Store', 'Dept'])['y']
        .mean()
    )
    return _means_to_dict(means), float(y_train.mean())


def _baseline_predict(X_train, y_train, X_test):
    means_dict, global_mean = _fit_baseline_means(X_train, y_train)
    return _baseline_vector(
        means_dict, global_mean, X_test['Store'].values, X_test['Dept'].values
    )


def _predict(artifact, X_scaled, X_df):
    """Score a saved artifact (direct model or residual = baseline + RF)."""
    model = artifact['model']
    if artifact.get('kind') == 'residual':
        base = _baseline_vector(
            artifact['baseline_means'],
            artifact['global_mean'],
            X_df['Store'].values,
            X_df['Dept'].values,
        )
        return base + model.predict(X_scaled)
    return model.predict(X_scaled)


def _train_and_save(model_choice):
    X_train, X_test, y_train, y_test, cutoff, _n_neg = _prepare_splits()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    means_dict, global_mean = _fit_baseline_means(X_train, y_train)

    if model_choice == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        artifact = {
            'model': model,
            'scaler': scaler,
            'feature_cols': FEATURE_COLS,
            'kind': 'direct',
        }
    elif model_choice == "Random Forest Regressor":
        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(X_train_scaled, y_train)
        artifact = {
            'model': model,
            'scaler': scaler,
            'feature_cols': FEATURE_COLS,
            'kind': 'direct',
        }
    else:
        # Residual RF: learn deviations from Store×Dept mean
        y_base_train = _baseline_vector(
            means_dict, global_mean, X_train['Store'].values, X_train['Dept'].values
        )
        resid = y_train.values - y_base_train
        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(X_train_scaled, resid)
        artifact = {
            'model': model,
            'scaler': scaler,
            'feature_cols': FEATURE_COLS,
            'kind': 'residual',
            'baseline_means': means_dict,
            'global_mean': global_mean,
        }

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(artifact, _model_path(model_choice))

    y_pred = _predict(artifact, X_test_scaled, X_test)
    info = {
        'model': model_choice,
        'kind': artifact['kind'],
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

    return artifact, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, cutoff


def _features_compatible(saved):
    if list(saved.get('feature_cols') or []) != FEATURE_COLS:
        return False
    if saved.get('kind') == 'residual':
        return 'baseline_means' in saved and 'global_mean' in saved
    return saved.get('kind', 'direct') == 'direct'


@st.cache_resource
def get_model_pipeline(model_choice):
    "Load model from disk if compatible, otherwise train and save."
    path = _model_path(model_choice)
    if os.path.exists(path):
        saved = joblib.load(path)
        if _features_compatible(saved):
            X_train, X_test, y_train, y_test, cutoff, _ = _prepare_splits()
            scaler = saved['scaler']
            return (
                saved,
                X_train,
                X_test,
                y_train,
                y_test,
                scaler.transform(X_train),
                scaler.transform(X_test),
                cutoff,
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
        "Forecast weekly department-level sales on a future hold-out, "
        "with a Store×Dept baseline as the bar to beat."
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
            "- **Date**: replaced by `Week` / `Month` to capture seasonality "
            "without leaking the absolute timestamp.\n"
            "- **MarkDown1–5**: sparse (only available post Nov 2011) and weakly "
            "correlated with sales in our analysis.\n"
            "- **Size**: strongly collinear with store identity.\n\n"
            "**Target cleaning**: rows with `Weekly_Sales < 0` (returns/adjustments) "
            "are dropped before training.",
            icon="ℹ️",
        )

    st.write("---")

    # ── 2. Model training ─────────────────────────────────────────────────────
    st.subheader("2 · Model Training")

    model_choice = st.selectbox("Select a model:", MODEL_CHOICES, index=2)
    st.caption(
        "**Linear Regression**: pedagogical foil (numeric IDs break linearity). "
        "**Random Forest**: direct sales model. "
        "**Residual Random Forest**: Store×Dept mean + RF on residuals "
        "(production pick when it beats the baseline)."
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
        artifact, X_train, X_test, y_train, y_test,
        X_train_scaled, X_test_scaled, cutoff,
    ) = pipeline
    model = artifact['model']
    scaler = artifact['scaler']

    if will_load:
        info_path = _info_path(model_choice)
        if os.path.exists(info_path):
            with open(info_path, encoding='utf-8') as f:
                info = json.load(f)
            st.success(
                f"Model loaded from disk, trained {info['trained_at']} "
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
            "Temporal 80/20 on unique weeks · negative sales dropped · "
            "`Type` A/B/C → 0/1/2 · `Week`/`Month` from Date · StandardScaler"
        )
        st.dataframe(
            pd.DataFrame(X_train_scaled[:5], columns=X_train.columns).round(3),
            width='stretch',
        )

    st.write("---")

    # ── 3. Evaluation ─────────────────────────────────────────────────────────
    st.subheader("3 · Model Evaluation")

    y_pred = _predict(artifact, X_test_scaled, X_test)
    metrics = _compute_metrics(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("R²", f"{metrics['R²']:.4f}")
    col2.metric("RMSE", f"${metrics['RMSE ($)']:,.0f}")
    col3.metric("MAE", f"${metrics['MAE ($)']:,.0f}")

    if model_choice == "Linear Regression":
        st.warning(
            "**Not a production pick**: Linear Regression is a counter-example. "
            "`Store` and `Dept` are categorical IDs treated as continuous values, "
            "so the model collapses. Prefer **Residual Random Forest**.",
            icon="⚠️",
        )
    elif model_choice == "Residual Random Forest":
        st.success(
            "This model predicts **baseline + residual**: identity captures the level, "
            "the forest learns holiday/seasonal and contextual deviations.",
            icon="✅",
        )

    with st.expander("What do these metrics mean?"):
        st.write("""
        - **R²**: proportion of sales variance explained by the model. 1.0 is perfect.
        - **RMSE**: typical prediction error in dollars (penalises large errors more).
        - **MAE**: average absolute error in dollars (more robust to outliers).
        - Metrics use a **future** hold-out (weeks after the cutoff), not a random split.
        """)

    st.write("")

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
        title=f"Actual vs Predicted ({model_choice}, temporal hold-out)",
        color_discrete_sequence=["#636EFA"],
    )
    fig.add_shape(
        type="line", x0=0, y0=0, x1=max_val, y1=max_val,
        line={"color": "red", "dash": "dash", "width": 1.5},
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, width='stretch')

    if model_choice in ("Random Forest Regressor", "Residual Random Forest"):
        importance_df = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "Importance": model.feature_importances_,
        }).sort_values("Importance")
        title = (
            "Residual Feature Importance (Random Forest)"
            if model_choice == "Residual Random Forest"
            else "Feature Importance (Random Forest)"
        )
        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title=title,
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
        "- **Baseline**: train-period mean per (Store, Dept). The bar to beat.\n"
        "- **Linear Regression**: pedagogical foil.\n"
        "- **Random Forest**: direct model (often loses to baseline).\n"
        "- **Residual Random Forest**: baseline + RF on residuals (target design).",
        icon="ℹ️",
    )

    rows = []
    y_base = _baseline_predict(X_train, y_train, X_test)
    rows.append({"Model": "Baseline (Store × Dept mean)", **_compute_metrics(y_test, y_base)})

    for choice in MODEL_CHOICES:
        with st.spinner(f"Evaluating {choice}…"):
            art, _, x_te, _, y_t, _, x_t_sc, _ = get_model_pipeline(choice)
        y_p = _predict(art, x_t_sc, x_te)
        rows.append({"Model": choice, **_compute_metrics(y_t, y_p)})

    comparison = pd.DataFrame(rows).set_index("Model")
    styled = (
        comparison.style
        .highlight_max(axis=0, subset=["R²"], color="#d4edda")
        .highlight_min(axis=0, subset=["RMSE ($)", "MAE ($)"], color="#d4edda")
    )
    st.dataframe(styled, width='stretch')

    base_r2 = comparison.loc["Baseline (Store × Dept mean)", "R²"]
    resid_r2 = comparison.loc["Residual Random Forest", "R²"]
    if resid_r2 > base_r2:
        st.success(
            f"**Residual Random Forest beats the baseline** "
            f"(R² {resid_r2:.4f} vs {base_r2:.4f} on the temporal hold-out).",
            icon="🏆",
        )

    st.write("---")

    # ── 5. Make a prediction ──────────────────────────────────────────────────
    st.subheader("5 · Make a Prediction")
    st.caption("Pick a real store/department and a week date; Week/Month are derived automatically.")

    observed_depts = sorted(X_train['Dept'].astype(int).unique().tolist())

    col1, col2 = st.columns(2)
    with col1:
        store = st.number_input("Store (1–45)", min_value=1, max_value=45, value=20)
        dept = st.selectbox("Department", observed_depts, index=observed_depts.index(1) if 1 in observed_depts else 0)
        is_holiday = st.selectbox(
            "Holiday week?", [False, True], format_func=lambda x: "Yes" if x else "No"
        )
        store_type = st.selectbox("Store type", ["A", "B", "C"])
        week_date = st.date_input("Week date", value=date(2012, 6, 22))
    with col2:
        temperature = st.number_input("Temperature (°F)", value=60.0)
        fuel_price = st.number_input("Fuel Price ($/gal)", value=3.50)
        cpi = st.number_input("CPI", value=210.0)
        unemployment = st.number_input("Unemployment (%)", value=8.0)

    week = int(pd.Timestamp(week_date).isocalendar().week)
    month = int(pd.Timestamp(week_date).month)
    st.caption(f"Derived calendar features: ISO week **{week}**, month **{month}**.")

    input_data = pd.DataFrame(
        [[
            store, dept, int(is_holiday), temperature, fuel_price,
            cpi, unemployment, TYPE_CODES[store_type], week, month,
        ]],
        columns=FEATURE_COLS,
    )
    input_scaled = scaler.transform(input_data)
    prediction = float(_predict(artifact, input_scaled, input_data)[0])

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
                {model_choice} · Store {store} · Dept {dept} · {week_date}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
