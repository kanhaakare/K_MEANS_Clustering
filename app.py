import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load files safely
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('data.scaling.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("Model or Scaler files not found! Make sure model.pkl and data.scaling.pkl are in the same folder.")

st.title("Health Diagnosis Predictor App")
st.write("Enter your clinical details to evaluate your health classification index.")

# Input fields matching training structure
age = st.number_input("Age", min_value=1, max_value=120, value=50)
bmi = st.number_input("BMI", min_value=5.0, max_value=60.0, value=25.0)
glucose = st.number_input("Glucose", min_value=10.0, max_value=300.0, value=90.0)
insulin = st.number_input("Insulin", min_value=0.1, max_value=100.0, value=6.0)
homa = st.number_input("HOMA", min_value=0.1, max_value=30.0, value=1.5)
leptin = st.number_input("Leptin", min_value=1.0, max_value=150.0, value=15.0)
adiponectin = st.number_input("Adiponectin", min_value=1.0, max_value=50.0, value=10.0)
resistin = st.number_input("Resistin", min_value=1.0, max_value=100.0, value=10.0)
mcp_1 = st.number_input("MCP-1", min_value=10.0, max_value=2000.0, value=400.0)

if st.button("Predict classification"):
    input_data = np.array([[age, bmi, glucose, insulin, homa, leptin, adiponectin, resistin, mcp_1]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    
    st.subheader(f"Prediction Outcome: Class {prediction}")
    if prediction == 1:
        st.success("Classification Result: Class 1 (Healthier metabolic profile indications)")
    else:
        st.warning("Classification Result: Class 2 (Higher risk metabolic profiling)")
