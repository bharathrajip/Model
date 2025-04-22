import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('disease_prediction_model.pkl')

st.title("AI Disease Prediction App")

# Feature list
features = ['Age', 'Gender', 'Blood_Pressure', 'Heart_Rate', 'Cholesterol',
            'Glucose', 'Smoking', 'Alcohol', 'Physical_Activity']

# Create input form
user_input = {}
for feature in features:
    user_input[feature] = st.number_input(f"Enter {feature}", value=0.0)

if st.button("Predict Disease"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Disease: {prediction}")
