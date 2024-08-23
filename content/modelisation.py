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

def modelisation():
    "Modeling page content"

    st.title("Weekly Sales Prediction")

    # Load the data
    st.subheader("Data Loading")
    try:
        data = pd.read_csv('data/merged_retail_data.csv')
        st.success("Data successfully loaded!", icon = "✅")
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return

    st.write("---")

    # Data Preprocessing
    st.subheader("Data Preprocessing")
    
    # Select features and target variable
    st.write("Selecting relevant features and target variable.")
    features = data[['Store', 'Dept', 'IsHoliday_x', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Type']]
    target = data['Weekly_Sales']
    
    # Encoding categorical variables
    features['Type'] = features['Type'].astype('category').cat.codes
    
    # Train-test split
    st.write("Splitting the data into training and testing sets.")
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Feature scaling
    st.write("Applying feature scaling to standardize the numerical features.")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    st.write("---")

    # Model Selection and Training
    st.subheader("Model Selection and Training")
    
    # Option to select model
    model_choice = st.selectbox("Choose a model to train:", ("Linear Regression", "Random Forest Regressor"))
    
    if model_choice == "Linear Regression":
        model = LinearRegression()
    elif model_choice == "Random Forest Regressor":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Train the model
    st.write(f"Training the {model_choice} model...")
    model.fit(X_train, y_train)
    st.success(f"{model_choice} model trained successfully!")
    
    st.write("---")

    # Model Evaluation
    st.subheader("Model Evaluation")
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    st.write(f"**Mean Squared Error (MSE):** {mse}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse}")
    st.write(f"**R-squared (R2):** {r2}")
    
    st.write("---")

    # Predictions
    st.subheader("Make Predictions")
    
    # Input features for prediction
    st.write("Input the following features to predict the Weekly Sales:")
    store = st.number_input("Store", min_value=1, max_value=45)
    dept = st.number_input("Department", min_value=1)
    is_holiday = st.selectbox("Is Holiday?", [0, 1])
    temperature = st.number_input("Temperature")
    fuel_price = st.number_input("Fuel Price")
    cpi = st.number_input("CPI")
    unemployment = st.number_input("Unemployment")
    store_type = st.selectbox("Store Type", [0, 1, 2])  # Assuming these are encoded as 0, 1, 2
    
    input_data = np.array([[store, dept, is_holiday, temperature, fuel_price, cpi, unemployment, store_type]])
    input_data = scaler.transform(input_data)
    
    prediction = model.predict(input_data)
    
    st.write(f"**Predicted Weekly Sales:** ${prediction[0]:,.2f}")

