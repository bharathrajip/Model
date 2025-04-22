
import streamlit as st
import pandas as pd
import joblib
import pickle

# Load model and feature names
model = joblib.load("disease_prediction_model.pkl")
feature_names = pickle.load(open("feature_names.pkl", "rb"))

st.title("AI-Powered Disease Prediction App")
st.subheader("Select your symptoms:")

user_input = {}
for feature in feature_names:
    user_input[feature] = st.checkbox(feature.replace("_", " ").capitalize())

if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Disease: {prediction}")
