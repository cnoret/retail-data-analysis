"""
Page : Modeling and prediction
"""

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

FEATURE_COLS = [
    'Store', 'Dept', 'IsHoliday_x', 'Temperature',
    'Fuel_Price', 'CPI', 'Unemployment', 'Type',
]


@st.cache_data
def load_model_data():
    return pd.read_csv('data/merged_retail_data.csv')


@st.cache_resource
def get_model_pipeline(model_choice):
    data = pd.read_csv('data/merged_retail_data.csv')
    X = data[FEATURE_COLS].copy()
    y = data['Weekly_Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train.loc[:, 'Type'] = X_train['Type'].astype('category').cat.codes
    X_test.loc[:, 'Type'] = X_test['Type'].astype('category').cat.codes
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    if model_choice == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled


def modelisation():
    "Modeling page content"

    st.title("Weekly Sales Prediction")
    st.info("Let's clean, transform, and prepare the dataset for analysis !", icon="🔧")

    st.write("---")

    # Load the data
    st.subheader("Loading data")
    try:
        data = load_model_data()
        st.success("merged_retail_data.csv successfully loaded!", icon="✅")
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return

    st.write("---")

    ## Data Preprocessing
    st.subheader("Data Preprocessing")

    st.write("""
    **Why some features were not chosen:**
    - **Date**: The `Date` feature was not directly used in the model because it is a temporal feature that doesn't contribute directly to the prediction of sales. However, seasonal effects could be captured indirectly by features like `IsHoliday`, `Temperature`, and `Fuel_Price`.
    - **MarkDown1-5**: These features represent promotional markdowns, but they were excluded as they are not always available and might introduce noise rather than improving the model's performance. In specific scenarios, however, these could be revisited for a different modeling approach focusing on promotional impacts.
    """)

    st.write("**Selecting relevant features :**")
    st.dataframe(data[FEATURE_COLS].head())

    st.write("**Target variable :**")
    st.write(data['Weekly_Sales'])

    st.write("---")

    # Model Selection and Training
    st.subheader("Model Selection and Training")

    model_choice = st.selectbox("Choose a model to train :", ("Linear Regression", "Random Forest Regressor"))

    if model_choice == "Random Forest Regressor":
        label = f"Training the {model_choice} model... (~ 2 minutes)"
    else:
        label = f"Training the {model_choice} model..."
    with st.spinner(label):
        pipeline = get_model_pipeline(model_choice)
    model, scaler, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = pipeline

    st.success(f"{model_choice} model trained successfully!", icon="✅")

    st.info("**Splitting the data into training and testing sets (80% / 20%)**", icon="🔧")
    st.write(f"**Training set shape:** {X_train.shape}")
    st.write(f"**Testing set shape:** {X_test.shape}")
    st.write("**Training features sample:**")
    st.dataframe(X_train[:5])

    st.write("---")

    st.info("**Encoding categorical variables**", icon="🔧")
    st.write("**Encoded training features sample:**")
    st.dataframe(X_train[:5])

    st.write("---")

    st.info("**Applying feature scaling to standardize the numerical features.**", icon="🔧")
    st.write("**Scaled training features sample:**")
    st.dataframe(pd.DataFrame(X_train_scaled, columns=X_train.columns).head())

    st.write("---")

    # Model Evaluation
    st.subheader("Model Evaluation")

    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.write(f"**R-squared (R²):** {r2:.2f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")

    st.info("""
    **Explanation of metrics:**
    - **Root Mean Squared Error (RMSE):** The RMSE is the square root of the MSE and provides an error metric in the same units as the target variable (sales).
    - **R-squared (R²):** R² represents the proportion of the variance in the dependent variable (weekly sales) that is predictable from the independent variables. An R² close to 1 indicates that the model explains most of the variance in the outcome.
    """, icon='✨')

    st.write("---")

    # Predictions
    st.subheader('Make your own "Weekly_Sales" predictions !')

    st.write("Input the following features to predict the Weekly Sales:")
    store = st.number_input("Store", min_value=1, max_value=45)
    dept = st.number_input("Department", min_value=1)
    is_holiday = st.selectbox("Is Holiday?", [0, 1])
    temperature = st.number_input("Temperature")
    fuel_price = st.number_input("Fuel Price")
    cpi = st.number_input("CPI")
    unemployment = st.number_input("Unemployment")
    store_type = st.selectbox("Store Type", [0, 1, 2])

    input_data = pd.DataFrame(
        [[store, dept, is_holiday, temperature, fuel_price, cpi, unemployment, store_type]],
        columns=FEATURE_COLS,
    )
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)

    st.write("**Selected user data:**")
    st.dataframe(input_data)
    st.write("**Scaled user data:**")
    st.dataframe(pd.DataFrame(input_data_scaled, columns=X_train.columns).head())
    st.success(f"**Predicted Weekly Sales:** ${prediction[0]:,.2f}", icon="🤖")
