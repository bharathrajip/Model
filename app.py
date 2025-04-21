import streamlit as st
import numpy as np
import joblib
import pandas as pd  # Required to create a DataFrame for model input

# Load the trained machine learning model
model = joblib.load("disease_prediction_model.pkl")

# List of feature names (make sure these match your model's training features)
feature_names = [
    "Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4", "Symptom_5",
    "Symptom_6", "Symptom_7", "Symptom_8", "Symptom_9", "Symptom_10",
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing",
    "shivering", "chills", "joint_pain", "stomach_pain", "acidity"
    # Add any remaining features used in your model
]

# Streamlit app title
st.title("AI-Powered Disease Prediction")

# Instructions
st.write("Please enter values for each symptom/feature below to get a disease prediction.")

# Collect user input for all features
input_data = []
for feature in feature_names:
    value = st.number_input(f"{feature}", step=1.0)
    input_data.append(value)

# Prediction section
if st.button("Predict Disease"):
    # Convert input list to DataFrame with correct column names
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # Predict disease using the trained model
    prediction = model.predict(input_df)[0]

    # Display the prediction result
    st.success(f"Predicted Disease ID: {prediction}")
