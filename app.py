
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Health Diagnosis Predictor",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #f0f7ff, #f8faff, #edfffb);
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 1.7rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

.hero {
    background: linear-gradient(120deg, #ffffff, #e9f5ff);
    padding: 27px 30px;
    border-radius: 22px;
    border: 1px solid #dceafe;
    box-shadow: 0 7px 25px rgba(48, 94, 160, 0.08);
    margin-bottom: 22px;
}

.hero h1 {
    color: #15346b;
    font-size: 34px;
    font-weight: 800;
    margin: 0;
}

.hero p {
    color: #62789d;
    font-size: 15px;
    margin: 9px 0 0 0;
}

.section-card {
    background: rgba(255,255,255,0.90);
    border: 1px solid #dce9fb;
    border-radius: 20px;
    padding: 23px;
    box-shadow: 0 5px 20px rgba(37, 77, 130, 0.06);
    margin-bottom: 18px;
}

.section-title {
    color: #173b75;
    font-size: 22px;
    font-weight: 750;
    margin-bottom: 4px;
}

.section-subtitle {
    color: #7385a2;
    font-size: 13px;
    margin-bottom: 18px;
}

div[data-testid="stNumberInput"] label {
    color: #244577;
    font-weight: 600;
}

div[data-testid="stNumberInput"] input {
    border-radius: 10px;
}

div[data-testid="stNumberInput"] input:focus {
    border-color: #4a9df5;
}

.stButton > button {
    width: 100%;
    min-height: 52px;
    background: linear-gradient(100deg, #39b7ed, #627cf6, #a77cf6);
    color: white;
    border: none;
    border-radius: 12px;
    font-size: 17px;
    font-weight: 700;
    box-shadow: 0 5px 14px rgba(94, 116, 239, 0.22);
    transition: 0.2s ease;
}

.stButton > button:hover {
    color: white;
    border: none;
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(94, 116, 239, 0.30);
}

.result-card {
    border-radius: 18px;
    padding: 24px;
    margin-top: 20px;
}

.healthy {
    background: linear-gradient(120deg, #edfff5, #d9f8eb);
    border: 1px solid #b9eed2;
}

.risk {
    background: linear-gradient(120deg, #fff5f1, #ffebe8);
    border: 1px solid #ffd1c7;
}

.result-label {
    color: #45617d;
    font-size: 15px;
    font-weight: 600;
}

.result-number {
    font-size: 36px;
    font-weight: 800;
    margin: 6px 0;
}

.footer {
    color: #8191aa;
    font-size: 12px;
    text-align: center;
    padding: 20px 0 5px;
}

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #e3f2ff, #f4f6ff, #e3faf5);
    border-right: 1px solid #d7e7f8;
}

div[data-testid="stSidebar"] h2,
div[data-testid="stSidebar"] h3 {
    color: #183d79;
}

@media (max-width: 768px) {
    .hero h1 {
        font-size: 25px;
    }
    .section-card {
        padding: 15px;
    }
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL AND SCALER ----------------
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("data.scaling.pkl", "rb") as file:
        scaler = pickle.load(file)

    return model, scaler


model = None
scaler = None
artifacts_error = None

try:
    if not Path("model.pkl").exists():
        raise FileNotFoundError("model.pkl is missing.")

    if not Path("data.scaling.pkl").exists():
        raise FileNotFoundError("data.scaling.pkl is missing.")

    model, scaler = load_artifacts()

except Exception as error:
    artifacts_error = str(error)


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>💙 Health Diagnosis Predictor</h1>
    <p>Better Insights &nbsp; • &nbsp; Smarter Health Screening
    &nbsp; • &nbsp; Healthier Tomorrow</p>
</div>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 👋 Welcome!")
    st.write(
        "Explore your clinical measurements with a "
        "machine-learning classification tool."
    )

    st.divider()

    st.markdown("### 🛡️ Early Insights")
    st.caption("Explore patterns in your clinical measurements.")

    st.markdown("### 🧠 Machine Learning")
    st.caption("A trained model classifies the submitted measurements.")

    st.markdown("### 📋 Simple & Easy")
    st.caption("Enter your values and view the model's output.")

    st.divider()

    st.markdown("### 💚 Healthy Habits")
    st.write("🥗 Balanced nutrition")
    st.write("🏃 Regular physical activity")
    st.write("💧 Adequate hydration")
    st.write("🩺 Routine medical checkups")

    st.divider()

    st.caption("Your information is processed by this app.")
    st.caption(
        "Avoid entering identifying information. "
        "This app does not provide a medical diagnosis."
    )


# ---------------- ERROR HANDLING ----------------
if artifacts_error:
    st.error(
        "⚠️ The model or scaler could not be loaded. "
        "Please check your project files."
    )
    st.code(artifacts_error)
    st.stop()


# ---------------- INPUT SECTION ----------------
st.markdown("""
<div class="section-card">
    <div class="section-title">🩺 Enter Your Clinical Details</div>
    <div class="section-subtitle">
        Fill in the measurements below using values and units
        consistent with your model's training dataset.
    </div>
</div>
""", unsafe_allow_html=True)


# Personal and body measurements
st.markdown("### 👤 Basic Measurements")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age (years)", min_value=1, max_value=120, value=50
    )

with col2:
    bmi = st.number_input(
        "BMI", min_value=5.0, max_value=60.0,
        value=25.0, step=0.1, format="%.1f"
    )

with col3:
    glucose = st.number_input(
        "Glucose", min_value=10.0, max_value=300.0,
        value=90.0, step=1.0
    )


# Metabolic measurements
st.markdown("### 🧬 Metabolic Measurements")

col1, col2, col3 = st.columns(3)

with col1:
    insulin = st.number_input(
        "Insulin", min_value=0.1, max_value=100.0,
        value=6.0, step=0.1
    )

with col2:
    homa = st.number_input(
        "HOMA", min_value=0.1, max_value=30.0,
        value=1.5, step=0.1
    )

with col3:
    leptin = st.number_input(
        "Leptin", min_value=1.0, max_value=150.0,
        value=15.0, step=0.5
    )


# Additional biomarkers
st.markdown("### 🔬 Additional Biomarkers")

col1, col2, col3 = st.columns(3)

with col1:
    adiponectin = st.number_input(
        "Adiponectin", min_value=1.0, max_value=50.0,
        value=10.0, step=0.5
    )

with col2:
    resistin = st.number_input(
        "Resistin", min_value=1.0, max_value=100.0,
        value=10.0, step=0.5
    )

with col3:
    mcp_1 = st.number_input(
        "MCP-1", min_value=10.0, max_value=2000.0,
        value=400.0, step=10.0
    )


# ---------------- PREDICTION ----------------
st.write("")

predict_clicked = st.button(
    "💙  Predict Classification",
    use_container_width=True
)

if predict_clicked:
    try:
        feature_names = [
            "Age", "BMI", "Glucose", "Insulin", "HOMA",
            "Leptin", "Adiponectin", "Resistin", "MCP-1"
        ]

        input_data = pd.DataFrame(
            [[
                age, bmi, glucose, insulin, homa,
                leptin, adiponectin, resistin, mcp_1
            ]],
            columns=feature_names
        )

        # Apply the same fitted scaler used during training.
        scaled_data = scaler.transform(input_data)

        prediction = model.predict(scaled_data)[0]

        st.markdown("---")
        st.markdown("## 📊 Your Prediction Result")

        # Preserve the interpretation used by your original app.
        if prediction == 1:
            st.markdown("""
            <div class="result-card healthy">
                <div class="result-label">PREDICTION OUTCOME</div>
                <div class="result-number" style="color:#159765">
                    ✅ Class 1
                </div>
                <div style="color:#18764f;font-weight:600">
                    Healthier metabolic profile indications
                </div>
                <p style="color:#416c5c;margin-top:12px">
                    The model assigned your measurements to Class 1.
                    Continue following healthy habits and consult a
                    healthcare professional for medical interpretation.
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="result-card risk">
                <div class="result-label">PREDICTION OUTCOME</div>
                <div class="result-number" style="color:#d85c48">
                    ⚠️ Class {prediction}
                </div>
                <div style="color:#a8483b;font-weight:600">
                    Review your metabolic health with a professional
                </div>
                <p style="color:#865950;margin-top:12px">
                    The model assigned your measurements to Class
                    {prediction}. This result does not confirm a disease
                    or establish your individual level of risk.
                    Discuss your measurements with a healthcare
                    professional.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Optional model probability, if supported.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(scaled_data)[0]

            st.markdown("### 📈 Model Class Probabilities")

            probability_data = pd.DataFrame({
                "Class": [
                    str(label) for label in model.classes_
                ],
                "Model probability": probabilities
            })

            st.bar_chart(
                probability_data.set_index("Class"),
                y="Model probability"
            )

            st.caption(
                "Probabilities reflect the model's output, not a "
                "clinically validated probability of disease."
            )

        # Show the submitted values
        with st.expander("🔎 Review Your Entered Measurements"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as error:
        st.error(
            "Prediction could not be completed. Check that the "
            "input features and preprocessing match model training."
        )
        st.caption(f"Technical details: {error}")


# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    💙 Health Diagnosis Predictor &nbsp; | &nbsp;
    Machine Learning &nbsp; | &nbsp;
    Data-Driven Insights
    <br><br>
    For educational and informational purposes only.
    Not a substitute for professional medical advice.
</div>
""", unsafe_allow_html=True)
