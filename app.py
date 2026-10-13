import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Health Diagnosis Predictor",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================
# 2. CUSTOM CSS
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: linear-gradient(135deg, #f0f7ff, #f8faff, #edfffb); }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1200px; }

/* Hero */
.hero {
    background: linear-gradient(120deg, #ffffff, #e4f2ff);
    padding: 28px 32px;
    border-radius: 22px;
    border: 1px solid #dceafe;
    box-shadow: 0 7px 25px rgba(48, 94, 160, 0.08);
    margin-bottom: 22px;
}
.hero h1 { color: #15346b; font-size: 34px; font-weight: 800; margin: 0; }
.hero p  { color: #62789d; font-size: 15px; margin: 9px 0 0; }

/* The form itself becomes the card */
[data-testid="stForm"] {
    background: rgba(255,255,255,0.95);
    border: 1px solid #dce9fb;
    border-radius: 20px;
    padding: 28px 30px 24px;
    box-shadow: 0 5px 20px rgba(37,77,130,0.06);
}

.form-title    { color: #173b75; font-size: 22px; font-weight: 750; margin: 0 0 2px; }
.form-subtitle { color: #7385a2; font-size: 13px; margin-bottom: 8px; }

.group-label {
    display: flex; align-items: center; gap: 10px;
    color: #244577; font-size: 15px; font-weight: 700;
    margin: 20px 0 6px;
}
.group-label::after {
    content: ""; flex: 1; height: 1px; background: #e3ecf9;
}

div[data-testid="stNumberInput"] label p { color: #244577; font-weight: 600; font-size: 13px; }
div[data-testid="stNumberInput"] input   { border-radius: 10px; }
div[data-testid="stNumberInput"] > div > div { border-radius: 10px; }

/* Submit button */
[data-testid="stFormSubmitButton"] button {
    width: 100%;
    min-height: 54px;
    background: linear-gradient(100deg, #39b7ed, #627cf6, #a77cf6);
    color: white;
    border: none;
    border-radius: 12px;
    font-size: 17px;
    font-weight: 700;
    box-shadow: 0 5px 14px rgba(94,116,239,0.25);
    transition: 0.2s ease;
}
[data-testid="stFormSubmitButton"] button:hover {
    color: white; border: none;
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(94,116,239,0.32);
}
[data-testid="stFormSubmitButton"] button p { font-weight: 700; }

/* Result cards */
.result-card { border-radius: 18px; padding: 24px 28px; margin-top: 18px; }
.healthy { background: linear-gradient(120deg, #edfff5, #d9f8eb); border: 1px solid #b9eed2; }
.risk    { background: linear-gradient(120deg, #fff5f1, #ffebe8); border: 1px solid #ffd1c7; }
.result-label  { color: #45617d; font-size: 13px; font-weight: 700; letter-spacing: .08em; }
.result-number { font-size: 34px; font-weight: 800; margin: 6px 0; }

/* Probability bars */
.prob-card {
    background: #fff; border: 1px solid #dce9fb; border-radius: 18px;
    padding: 22px 26px; margin-top: 16px;
}
.prob-title { color: #173b75; font-size: 18px; font-weight: 700; margin-bottom: 14px; }
.prob-row   { display: flex; align-items: center; gap: 14px; margin: 10px 0; }
.prob-name  { width: 80px; color: #244577; font-weight: 600; font-size: 14px; }
.prob-track { flex: 1; height: 14px; background: #e6edfa; border-radius: 8px; overflow: hidden; }
.prob-fill  { height: 100%; border-radius: 8px;
              background: linear-gradient(90deg, #39b7ed, #627cf6); }
.prob-pct   { width: 60px; text-align: right; color: #244577; font-size: 14px; font-weight: 600; }
.prob-note  { color: #7385a2; font-size: 12px; margin-top: 10px; }

.footer { color: #8191aa; font-size: 12px; text-align: center; padding: 22px 0 5px; }

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #e3f2ff, #f4f6ff, #e3faf5);
    border-right: 1px solid #d7e7f8;
}
div[data-testid="stSidebar"] h2, div[data-testid="stSidebar"] h3 { color: #183d79; }

@media (max-width: 768px) {
    .hero h1 { font-size: 25px; }
    [data-testid="stForm"] { padding: 16px; }
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


try:
    if not Path("model.pkl").is_file():
        raise FileNotFoundError("model.pkl is missing.")
    if not Path("data.scaling.pkl").is_file():
        raise FileNotFoundError("data.scaling.pkl is missing.")
    model, scaler = load_artifacts()
except Exception as error:
    st.error("Unable to load the model or scaler.")
    st.exception(error)
    st.stop()


# =====================================================
# 4. HEADER + SIDEBAR
# =====================================================
st.markdown("""
<div class="hero">
    <h1>💙 Health Diagnosis Predictor</h1>
    <p>Better Insights &nbsp; • &nbsp; Smarter Health Screening
    &nbsp; • &nbsp; Healthier Tomorrow</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 👋 Welcome!")
    st.write("Explore clinical measurements using a machine-learning "
             "classification model.")
    st.divider()
    st.markdown("### 🛡️ Health Insights")
    st.caption("Explore patterns in clinical measurements.")
    st.markdown("### 🧠 Machine Learning")
    st.caption("Obtain a prediction from your trained model.")
    st.markdown("### 📋 Easy to Use")
    st.caption("Enter values and view your model's output.")
    st.divider()
    st.markdown("### 💚 Healthy Habits")
    st.write("🥗 Balanced nutrition")
    st.write("🏃 Regular physical activity")
    st.write("💧 Adequate hydration")
    st.write("🩺 Routine medical checkups")
    st.divider()
    st.caption("Educational use only. Avoid entering identifying "
               "personal information.")


# =====================================================
# 5. INPUT FORM (single card)
# =====================================================
FEATURE_NAMES = ["age", "bmi", "glucose", "insulin", "homa",
                 "leptin", "adiponectin", "resistin", "mcp_1"]

with st.form("clinical_form", border=False):
    st.markdown("""
        <div class="form-title">🩺 Enter Your Clinical Details</div>
        <div class="form-subtitle">Enter all nine measurements using the same
        units and definitions used by your training dataset.</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="group-label">👤 Basic Measurements</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="large")
    age = c1.number_input("Age (years)", 1, 120, 50)
    bmi = c2.number_input("BMI", 5.0, 60.0, 25.0, step=0.1, format="%.1f")
    glucose = c3.number_input("Glucose", 10.0, 300.0, 90.0, step=1.0)

    st.markdown('<div class="group-label">🧬 Metabolic Measurements</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="large")
    insulin = c1.number_input("Insulin", 0.1, 100.0, 6.0, step=0.1)
    homa = c2.number_input("HOMA", 0.1, 30.0, 1.5, step=0.1)
    leptin = c3.number_input("Leptin", 1.0, 150.0, 15.0, step=0.5)

    st.markdown('<div class="group-label">🔬 Additional Biomarkers</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="large")
    adiponectin = c1.number_input("Adiponectin", 1.0, 50.0, 10.0, step=0.5)
    resistin = c2.number_input("Resistin", 1.0, 100.0, 10.0, step=0.5)
    mcp_1 = c3.number_input("MCP-1", 10.0, 2000.0, 400.0, step=10.0)

    st.write("")
    submitted = st.form_submit_button("💙 Predict Classification",
                                      use_container_width=True)

input_data = pd.DataFrame(
    [[age, bmi, glucose, insulin, homa, leptin, adiponectin, resistin, mcp_1]],
    columns=FEATURE_NAMES,
)


# =====================================================
# 6. PREDICTION
# =====================================================
if submitted:
    try:
        n_expected = getattr(scaler, "n_features_in_", None)
        if n_expected is not None and n_expected != len(FEATURE_NAMES):
            raise ValueError(
                f"The scaler expects {n_expected} features, "
                f"but the app supplies {len(FEATURE_NAMES)}."
            )

        scaler_features = None
        if hasattr(scaler, "feature_names_in_"):
            scaler_features = list(scaler.feature_names_in_)
            missing = [n for n in scaler_features if n not in input_data.columns]
            if missing:
                raise ValueError(f"Missing scaler features: {missing}")
            scaler_input = input_data[scaler_features]
        else:
            scaler_input = input_data.to_numpy(dtype=float)

        scaled = scaler.transform(scaler_input)

        if hasattr(model, "feature_names_in_"):
            model_features = list(model.feature_names_in_)
            if len(model_features) != scaled.shape[1]:
                raise ValueError("The model and scaler expect different "
                                 "numbers of features. Check training.")
            pred_input = pd.DataFrame(
                scaled, columns=scaler_features or FEATURE_NAMES
            )
            missing = [n for n in model_features if n not in pred_input.columns]
            if missing:
                raise ValueError("The model expects different feature names: "
                                 f"{missing}. Check the training pipeline.")
            pred_input = pred_input[model_features]
        else:
            pred_input = scaled

        prediction = model.predict(pred_input)[0]

        st.markdown("## 📊 Prediction Result")

        # NOTE: assumes class 1 = Healthy, otherwise Patient.
        # Confirm this mapping against your training dataset.
        if str(prediction) == "Healthy":
            st.markdown(f"""
            <div class="result-card healthy">
                <div class="result-label">PREDICTION OUTCOME</div>
                <div class="result-number" style="color:#159765">✅ Healthy Person</div>
                <div style="color:#18764f;font-weight:600">Model predicted Class {prediction}</div>
                <p style="color:#416c5c;margin-top:12px">
                    The model assigned your measurements to Class {prediction}.
                    This is not a medical diagnosis. Consult a healthcare
                    professional for interpretation of your results.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card risk">
                <div class="result-label">PREDICTION OUTCOME</div>
                <div class="result-number" style="color:#d85c48">⚠️ Patient</div>
                <div style="color:#a8483b;font-weight:600">Model predicted Class {prediction}</div>
                <p style="color:#865950;margin-top:12px">
                    The model assigned your measurements to Class {prediction}.
                    This does not confirm a disease. Please consult a
                    healthcare professional for further evaluation.
                </p>
            </div>
            """, unsafe_allow_html=True)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(pred_input)[0]
            rows = "".join(
                f"""<div class="prob-row">
                        <div class="prob-name">Class {label}</div>
                        <div class="prob-track"><div class="prob-fill"
                             style="width:{p*100:.1f}%"></div></div>
                        <div class="prob-pct">{p*100:.1f}%</div>
                    </div>"""
                for label, p in zip(model.classes_, probs)
            )
            st.markdown(f"""
            <div class="prob-card">
                <div class="prob-title">📈 Model Class Probabilities</div>
                {rows}
                <div class="prob-note">These probabilities describe the model
                output; they are not validated clinical risk estimates.</div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("🔎 Review Entered Measurements"):
            st.dataframe(input_data, use_container_width=True, hide_index=True)

    except Exception as error:
        st.error("❌ Prediction could not be completed.")
        st.exception(error)
        st.info("Check the feature names, feature order, scaler, and model "
                "training pipeline.")


# =====================================================
# 7. FOOTER
# =====================================================
st.markdown("""
<div class="footer">
    💙 Health Diagnosis Predictor &nbsp; | &nbsp; Machine Learning
    &nbsp; | &nbsp; Data-Driven Insights
    <br><br>
    For educational purposes only. Not a substitute for professional
    medical advice or diagnosis.
</div>
""", unsafe_allow_html=True)
