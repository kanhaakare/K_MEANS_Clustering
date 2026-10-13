import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="Metabolic Health Predictor",
    page_icon="🧬",
    layout="wide"
)

# Injected CSS
st.markdown("\"\"\"<style>
    .main {
        background-color: #f9fbff;
    }
    h1 {
        color: #1e3d59;
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-weight: 700;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #17b978;
        margin-bottom: 15px;
    }
    .metric-title {
        font-size: 14px;
        color: #8c8c8c;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1e3d59;
    }
</style>\"\"\", unsafe_allow_html=True)

try:
    with open('model.pkl', 'rb') as f_in:
        model = pickle.load(f_in)
    with open('data.scaling.pkl', 'rb') as f_sc:
        scaler = pickle.load(f_sc)
except FileNotFoundError:
    st.error("⚠️ Model or Scaler files not found! Make sure model.pkl and data.scaling.pkl are in the same directory.")

col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("<h1 style='font-size: 60px; margin: 0;'>🧬</h1>", unsafe_allow_html=True)
with col2:
    st.title("Metabolic Health Diagnostic Dashboard")
    st.write("Analyze and predict patient metabolic health indexes using our optimized K-Nearest Neighbors model.")

st.markdown("---")

st.sidebar.header("🔬 Clinical Measurements Input")
st.sidebar.write("Adjust patient variables below:")

age = st.sidebar.slider("Age (years)", 1, 120, 50)
bmi = st.sidebar.slider("Body Mass Index (BMI)", 5.0, 60.0, 25.0, step=0.1)
glucose = st.sidebar.slider("Glucose Level (mg/dL)", 10.0, 300.0, 90.0, step=1.0)
insulin = st.sidebar.slider("Insulin Level (µU/mL)", 0.1, 100.0, 6.0, step=0.1)
homa = st.sidebar.slider("HOMA-IR (Homeostatic Model Assessment)", 0.1, 30.0, 1.5, step=0.1)
leptin = st.sidebar.slider("Leptin (ng/mL)", 1.0, 150.0, 15.0, step=0.1)
adiponectin = st.sidebar.slider("Adiponectin (µg/mL)", 1.0, 50.0, 10.0, step=0.1)
resistin = st.sidebar.slider("Resistin (ng/mL)", 1.0, 100.0, 10.0, step=0.1)
mcp_1 = st.sidebar.slider("MCP-1 (Monocyte Chemoattractant Protein-1)", 10.0, 2000.0, 400.0, step=10.0)

st.subheader("📊 Current Patient Profile Overview")
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    st.markdown(f'<div class="metric-card"><p class="metric-title">Age</p><p class="metric-value">{age} yrs</p></div>', unsafe_allow_html=True)
with m_col2:
    st.markdown(f'<div class="metric-card"><p class="metric-title">BMI</p><p class="metric-value">{bmi:.1f}</p></div>', unsafe_allow_html=True)
with m_col3:
    st.markdown(f'<div class="metric-card"><p class="metric-title">Glucose</p><p class="metric-value">{glucose:.1f} mg/dL</p></div>', unsafe_allow_html=True)
with m_col4:
    st.markdown(f'<div class="metric-card"><p class="metric-title">HOMA-IR</p><p class="metric-value">{homa:.2f}</p></div>', unsafe_allow_html=True)

st.markdown("<h3 style='margin-top: 30px;'>🔮 AI Diagnostic Classification</h3>", unsafe_allow_html=True)

if st.button("Run Diagnostic Assessment Model", use_container_width=True):
    # Reconstruct input features into a Pandas DataFrame with identical column sequence as the training features
    input_df = pd.DataFrame([{
        'age': age,
        'bmi': bmi,
        'glucose': glucose,
        'insulin': insulin,
        'homa': homa,
        'leptin': leptin,
        'adiponectin': adiponectin,
        'resistin': resistin,
        'mcp_1': mcp_1
    }])
    
    scaled_data = scaler.transform(input_df)
    prediction = model.predict(scaled_data)[0]
    
    st.markdown("### Diagnostic Results:")
    if prediction == 1:
        st.success("🎉 **Class 1 Detected**: Indicates a healthier metabolic and biological profile structure.")
    else:
        st.warning("⚠️ **Class 2 Detected**: Indicates a higher risk profile with metabolic and physiological strain tendencies. Further screening recommended.")
