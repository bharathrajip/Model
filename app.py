import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("disease_prediction_model.pkl")

# Define symptom input fields
st.title("AI Disease Prediction App")
st.write("Enter symptoms to predict possible disease")

# Example symptoms (you can replace or add more based on your dataset)
symptoms = [
    "fever", "cough", "headache", "nausea", "fatigue", "pain", "vomiting", "diarrhea", "sore_throat"
]

user_input = {}
for symptom in symptoms:
    user_input[symptom] = st.checkbox(symptom.capitalize())

# Predict button
if st.button("Predict"):
    input_data = pd.DataFrame([user_input])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Disease: {prediction}")
