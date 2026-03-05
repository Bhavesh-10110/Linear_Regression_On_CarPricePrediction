import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv("mercedes_benz_listings_cleaned.csv")
df = pd.DataFrame(data)

# Use only numeric features for a stable model
feature_cols = ['Year', 'Vehicle_Age', 'Is_AMG', 'Is_4MATIC', 'Mileage_Miles']
x = df[feature_cols]
y = df['Price_USD']

# Train model
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
model = LinearRegression()
model.fit(x_train, y_train)

# Streamlit UI
st.title(" Mercedes-Benz Price Predictor")
st.write("Enter the car details below to predict the price")

col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Year", min_value=2015, max_value=2026, value=2023)
    vehicle_age = st.number_input("Vehicle Age (years)", min_value=1, max_value=15, value=2)
    mileage_miles = st.number_input("Mileage (Miles)", min_value=0, max_value=150000, value=20000)

with col2:
    is_amg = st.selectbox("Is AMG?", options=["No", "Yes"])
    is_4matic = st.selectbox("Is 4MATIC?", options=["No", "Yes"])

if st.button(" Predict Price"):
    # Convert Yes/No to 1/0
    is_amg_val = 1 if is_amg == "Yes" else 0
    is_4matic_val = 1 if is_4matic == "Yes" else 0
    
    # Create input dataframe
    user_input = pd.DataFrame({
        'Year': [year],
        'Vehicle_Age': [vehicle_age],
        'Is_AMG': [is_amg_val],
        'Is_4MATIC': [is_4matic_val],
        'Mileage_Miles': [mileage_miles]
    })
    
    # Predict
    predicted_price = model.predict(user_input)[0]
    
    # Ensure price is reasonable (not negative)
    predicted_price = max(predicted_price, 5000)
    
    st.success(f" Predicted Price: ${predicted_price:,.2f}")

