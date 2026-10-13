
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Health Diagnosis Predictor",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# 2. CUSTOM CSS
# =====================================================
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
    margin: 9px 0 0;
}

.section-card {
    background: rgba(255,255,255,0.92);
    border: 1px solid #dce9fb;
    border-radius: 20px;
    padding: 23px;
    box-shadow: 0 5px 20px rgba(37,77,130,0.06);
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

.stButton > button {
    width: 100%;
    min-height: 52px;
    background: linear-gradient(100deg, #39b7ed, #627cf6, #a77cf6);
    color: white;
    border: none;
    border-radius: 12px;
    font-size: 17px;
    font-weight: 700;
    box-shadow: 0 5px 14px rgba(94,116,239,0.22);
    transition: 0.2s ease;
}

.stButton > button:hover {
    color: white;
    border: none;
    transform: translateY(-2px);
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


# =====================================================
# 3. LOAD MODEL AND SCALER
# =====================================================
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("data.scaling.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


model = None
scaler = None

try:
    if not Path("model.pkl").is_file():
        raise FileNotFoundError(
            "model.pkl was not found in the project folder."
        )

    if not Path("data.scaling.pkl").is_file():
        raise FileNotFoundError(
            "data.scaling.pkl was not found in the project folder."
        )

    model, scaler = load_artifacts()

except Exception as error:
    st.error("Unable to load the model or scaler.")
    st.exception(error)
    st.stop()


# =====================================================
# 4. HEADER
# =====================================================
st.markdown("""
<div class="hero">
    <h1>💙 Health Diagnosis Predictor</h1>
    <p>Better Insights &nbsp; • &nbsp; Smarter Health Screening
    &nbsp; • &nbsp; Healthier Tomorrow</p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# 5. SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("## 👋 Welcome!")
    st.write(
        "Explore clinical measurements with a "
        "machine-learning classification tool."
    )

    st.divider()

    st.markdown("### 🛡️ Health Insights")
    st.caption("Explore patterns in submitted measurements.")

    st.markdown("### 🧠 Machine Learning")
    st.caption("Use a trained model to obtain a classification.")

    st.markdown("### 📋 Easy to Use")
    st.caption("Enter your values and view the model output.")

    st.divider()

    st.markdown("### 💚 Healthy Habits")
    st.write("🥗 Balanced nutrition")
    st.write("🏃 Regular physical activity")
    st.write("💧 Adequate hydration")
    st.write("🩺 Routine medical checkups")

    st.divider()

    st.caption(
        "Educational tool only. Avoid entering personal "
        "identifying information."
    )


# =====================================================
# 6. CLINICAL INPUT FORM
# =====================================================
st.markdown("""
<div class="section-card">
    <div class="section-title">🩺 Enter Your Clinical Details</div>
    <div class="section-subtitle">
        Enter all nine measurements using the same units
        and feature definitions as your training dataset.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("### 👤 Basic Measurements")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age (years)",
        min_value=1,
        max_value=120,
        value=50
    )

with col2:
    bmi = st.number_input(
        "BMI",
        min_value=5.0,
        max_value=60.0,
        value=25.0,
        step=0.1,
        format="%.1f"
    )

with col3:
    glucose = st.number_input(
        "Glucose",
        min_value=10.0,
        max_value=300.0,
        value=90.0,
        step=1.0
    )

st.markdown("### 🧬 Metabolic Measurements")

col1, col2, col3 = st.columns(3)

with col1:
    insulin = st.number_input(
        "Insulin",
        min_value=0.1,
        max_value=100.0,
        value=6.0,
        step=0.1
    )

with col2:
    homa = st.number_input(
        "HOMA",
        min_value=0.1,
        max_value=30.0,
        value=1.5,
        step=0.1
    )

with col3:
    leptin = st.number_input(
        "Leptin",
        min_value=1.0,
        max_value=150.0,
        value=15.0,
        step=0.5
    )

st.markdown("### 🔬 Additional Biomarkers")

col1, col2, col3 = st.columns(3)

with col1:
    adiponectin = st.number_input(
        "Adiponectin",
        min_value=1.0,
        max_value=50.0,
        value=10.0,
        step=0.5
    )

with col2:
    resistin = st.number_input(
        "Resistin",
        min_value=1.0,
        max_value=100.0,
        value=10.0,
        step=0.5
    )

with col3:
    mcp_1 = st.number_input(
        "MCP-1",
        min_value=10.0,
        max_value=2000.0,
        value=400.0,
        step=10.0
    )


# =====================================================
# 7. PREPARE MODEL INPUT
# =====================================================
feature_names = [
    "Age",
    "BMI",
    "Glucose",
    "Insulin",
    "HOMA",
    "Leptin",
    "Adiponectin",
    "Resistin",
    "MCP-1"
]

values = [
    age,
    bmi,
    glucose,
    insulin,
    homa,
    leptin,
    adiponectin,
    resistin,
    mcp_1
]

input_data = pd.DataFrame(
    [values],
    columns=feature_names
)


# =====================================================
# 8. PREDICTION
# =====================================================
st.write("")

if st.button(
    "💙  Predict Classification",
    use_container_width=True
):
    try:
        # Validate the number of expected features.
        expected_features = getattr(
            scaler, "n_features_in_", None
        )

        if (
            expected_features is not None
            and expected_features != input_data.shape[1]
        ):
            raise ValueError(
                f"The scaler expects {expected_features} features, "
                f"but the app provides {input_data.shape[1]}. "
                "Check your model-training code."
            )

        # Use column names if the scaler was trained with them.
        if hasattr(scaler, "feature_names_in_"):
            scaler_features = list(scaler.feature_names_in_)

            missing = [
                name for name in scaler_features
                if name not in input_data.columns
            ]

            if missing:
                raise ValueError(
                    f"These scaler features are missing: {missing}. "
                    "Update the input names to match training."
                )

            scaler_input = input_data[scaler_features]
        else:
            # For scalers trained on NumPy arrays.
            scaler_input = input_data.to_numpy(dtype=float)

        scaled_data = scaler.transform(scaler_input)

        # Preserve feature names if the model expects them.
        if hasattr(model, "feature_names_in_"):
            model_features = list(model.feature_names_in_)

            if len(model_features) != scaled_data.shape[1]:
                raise ValueError(
                    "The model's expected feature count does not "
                    "match the scaled data."
                )

            prediction_input = pd.DataFrame(
                scaled_data,
                columns=model_features
            )
        else:
            prediction_input = scaled_data

        # Make the prediction.
        prediction = model.predict(prediction_input)[0]

        st.markdown("---")
        st.markdown("## 📊 Your Prediction Result")

        # NOTE:
        # Confirm the meaning of the labels from your dataset.
        # This app cannot determine clinical meaning by itself.
        if str(prediction) == "1":
            st.markdown("""
            <div class="result-card healthy">
                <div class="result-label">PREDICTION OUTCOME</div>
                <div class="result-number" style="color:#159765">
                    ✅ Class 1
                </div>
                <div style="color:#18764f;font-weight:600">
                    Model classification: Class 1
                </div>
                <p style="color:#416c5c;margin-top:12px">
                    Your submitted measurements were assigned
                    to Class 1 by the model. Please consult a
                    healthcare professional for interpretation.
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="result-card risk">
                <div class="result-label">PREDICTION OUTCOME</div>
                <div class="result-number" style="color:#d85c48">
                    📋 Class {prediction}
                </div>
                <div style="color:#a8483b;font-weight:600">
                    Model classification: Class {prediction}
                </div>
                <p style="color:#865950;margin-top:12px">
                    Your submitted measurements were assigned
                    to Class {prediction} by the model. This result
                    does not establish a diagnosis. Discuss your
                    measurements with a healthcare professional.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Display model probabilities if supported.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(
                prediction_input
            )[0]

            st.markdown("### 📈 Model Class Probabilities")

            probability_data = pd.DataFrame({
                "Class": [
                    str(label) for label in model.classes_
                ],
                "Probability": probabilities
            })

            st.bar_chart(
                probability_data.set_index("Class")
            )

            st.caption(
                "These are model probabilities, not validated "
                "clinical probabilities or a measure of disease risk."
            )

        # Show the input values.
        with st.expander("🔎 Review Entered Measurements"):
            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )

    except Exception as error:
        st.error("❌ Prediction could not be completed.")
        st.markdown(
            "### 🔍 Technical error details"
        )
        st.exception(error)

        st.info(
            "Check the feature count, feature order, column names, "
            "scaler and model training pipeline. The technical "
            "error above identifies the operation that failed."
        )


# =====================================================
# 9. FOOTER
# =====================================================
st.markdown("""
<div class="footer">
    💙 Health Diagnosis Predictor &nbsp; | &nbsp;
    Machine Learning &nbsp; | &nbsp; Data-Driven Insights
    <br><br>
    For educational purposes only. Not a substitute for
    professional medical advice or diagnosis.
</div>
""", unsafe_allow_html=True)

