import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
@st.cache_data  # Updated to use the new caching method
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
    column_names = ['MedInc', 'AveRooms', 'AveOccup', 'AveBedrms', 'HouseAge', 'MedHouseVal']  # Removed duplicates
    data = pd.read_csv(url, sep='\s+', header=None, names=column_names)  # Changed to use 'sep' instead of 'delim_whitespace'
    return data

# Load the data
data = load_data()

# Features and target variable
X = data.drop('MedHouseVal', axis=1)
y = data['MedHouseVal']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the RandomForest model
model = RandomForestRegressor()
model.fit(X_train_scaled, y_train)

# Save the trained model and scaler for deployment
joblib.dump(model, 'house_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Define the Streamlit app UI
st.title("House Price Prediction")
st.write("""
    This is a simple Streamlit app that uses a RandomForest model to predict house prices in California.
""")

# User input fields
medinc = st.number_input('Median Income', min_value=0.0, max_value=200000.0, value=3.0, step=0.1)
averooms = st.number_input('Average Rooms', min_value=0, max_value=10, value=6, step=1)
aveoccup = st.number_input('Average Occupants', min_value=0, max_value=10, value=3, step=1)
avebedrms = st.number_input('Average Bedrooms', min_value=0, max_value=10, value=3, step=1)
houseage = st.number_input('House Age', min_value=0, max_value=100, value=20, step=1)

# Creating an input dataframe for prediction
input_data = pd.DataFrame([[medinc, averooms, aveoccup, avebedrms, houseage]], 
                          columns=['MedInc', 'AveRooms', 'AveOccup', 'AveBedrms', 'HouseAge'])

# Preprocessing the input data
input_data_scaled = scaler.transform(input_data)

# Predicting the price
prediction = model.predict(input_data_scaled)

# Display the prediction
st.write(f"The predicted house price is ${prediction[0]:,.2f} thousand")