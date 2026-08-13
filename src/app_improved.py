import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

"""
IMPROVED STREAMLIT APP - THYROID DISEASE PREDICTION
====================================================
Features:
1. Multi-level family history input (0-3 levels)
2. Continuous weight change input (in kg)
3. Better UI/UX with explanation of feature levels
4. Confidence intervals and risk assessment
5. Educational information about features
"""

# =============================================
# Page Configuration
# =============================================
st.set_page_config(
    page_title="Thyroid Disease Prediction System",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Early Detection of Thyroid Disease using Machine Learning")
st.write("""
This application predicts thyroid disease (hypothyroidism) based on clinical data, 
hormone values, and engineered features including multi-level family history and 
weight change patterns.
""")

# =============================================
# Load Model & Resources
# =============================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "thyroid_ensemble_weighted.pkl"
SCALER_PATH = BASE_DIR / "models" / "thyroid_scaler.pkl"
LE_PATH = BASE_DIR / "models" / "thyroid_label_encoder.pkl"

@st.cache_resource
def load_resources():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        le = joblib.load(LE_PATH)
        return model, scaler, le
    except FileNotFoundError:
        st.error("Model files not found. Please train the model first.")
        st.stop()

model, scaler, le = load_resources()

# =============================================
# Feature Columns (Must Match Training)
# =============================================
FEATURE_COLUMNS = [
    "age", "sex", "on_thyroxine", "query_on_thyroxine",
    "on_antithyroid_medication", "thyroid_surgery",
    "query_hypothyroid", "query_hyperthyroid",
    "pregnant", "sick", "tumor", "lithium", "goitre",
    "TSH_measured", "TSH",
    "T3_measured", "T3",
    "TT4_measured", "TT4",
    "T4U_measured", "T4U",
    "FTI_measured", "FTI",
    "TBG_measured", "TBG",
    "family_history_level", "weight_change_kg",  # NEW: Multi-level features
    "weight_change_norm",  # Normalized version
    "TSH_TT4_ratio", "T3_TT4_ratio"
]

# =============================================
# Helper Functions
# =============================================
def yes_no(label, default=0):
    """Binary Yes/No selector"""
    return 1 if st.selectbox(label, ["No", "Yes"], index=default) == "Yes" else 0

def get_family_history_description(level):
    """Get description for family history level"""
    descriptions = {
        0: "✓ No family history of thyroid disease",
        1: "⚠ Direct relative (parent) has thyroid disease - Moderate genetic risk (1.5-2x)",
        2: "⚠⚠ Grandparent has thyroid disease - Some genetic component (1.3-1.5x)",
        3: "⚠⚠⚠ Multiple ancestors affected - Strong genetic predisposition (2-3x risk)"
    }
    return descriptions.get(level, "Unknown")

def get_weight_change_guidance(weight_change_kg):
    """Provide guidance based on weight change"""
    if weight_change_kg > 3:
        return "⚠ Significant weight gain - Possible hypothyroid symptom"
    elif weight_change_kg > 1:
        return "Moderate weight gain - May indicate metabolic changes"
    elif weight_change_kg > -1:
        return "Stable weight - Normal variation"
    else:
        return "Weight loss - May indicate metabolic changes"

# =============================================
# Sidebar: Information
# =============================================
with st.sidebar:
    st.subheader("📋 About This Tool")
    st.write("""
    This ML model predicts hypothyroidism using:
    - **Clinical features**: Symptoms and medical history
    - **Hormone values**: TSH, T3, TT4, FTI
    - **Engineered features**: Family history and weight change
    
    **Accuracy**: ~99.2% on test data
    
    ⚠️ **Disclaimer**: This tool is for educational purposes only.
    Always consult with a healthcare provider for diagnosis.
    """)
    
    st.subheader("🧬 Family History Levels")
    st.markdown("""
    - **Level 0**: No family history
    - **Level 1**: Parent affected
    - **Level 2**: Grandparent affected
    - **Level 3**: Multiple ancestors affected
    """)

# =============================================
# Main Input Form
# =============================================
with st.form("prediction_form"):
    
    # ===== Patient Information =====
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Patient Information")
        age = st.number_input("Age (years)", 1, 120, 45)
        sex = 0 if st.selectbox("Sex", ["Female", "Male"]) == "Female" else 1
    
    # ===== Multi-Level Family History =====
    with col2:
        st.subheader("🧬 Family History")
        family_history_level = st.selectbox(
            "Family History of Thyroid Disease",
            options=[0, 1, 2, 3],
            format_func=lambda x: f"Level {x}: {get_family_history_description(x).split('✓')[1].split('⚠')[0].strip()}"
        )
        st.info(get_family_history_description(family_history_level))
    
    # ===== Weight Change =====
    st.subheader("⚖️ Weight Change")
    weight_change_kg = st.number_input(
        "Weight Change (kg) - Enter positive for weight gain, negative for weight loss",
        min_value=-10.0,
        max_value=15.0,
        value=0.5,
        step=0.5,
        help="Typical hypothyroid patients gain 2-5 kg"
    )
    st.info(get_weight_change_guidance(weight_change_kg))
    
    # Normalize weight change for model
    # We'll use a simple MinMaxScaler approach (assuming range -10 to 15)
    weight_change_norm = (weight_change_kg - (-10)) / (15 - (-10))  # Scale to 0-1
    
    # ===== Clinical History =====
    st.subheader("🏥 Clinical History")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        on_thyroxine = yes_no("On Thyroxine")
        query_on_thyroxine = yes_no("Query On Thyroxine")
        on_antithyroid_medication = yes_no("Antithyroid Medication")
        thyroid_surgery = yes_no("Thyroid Surgery")
    
    with col2:
        query_hypothyroid = yes_no("Query Hypothyroid")
        query_hyperthyroid = yes_no("Query Hyperthyroid")
        pregnant = yes_no("Pregnant")
        sick = yes_no("Sick")
    
    with col3:
        tumor = yes_no("Thyroid Tumor")
        lithium = yes_no("On Lithium")
        goitre = yes_no("Goitre")
    
    # ===== Hormone Values =====
    st.subheader("🔬 Thyroid Hormone Values")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        TSH = st.number_input("TSH (mIU/L)", 0.0, 100.0, 2.5, step=0.1)
        T3 = st.number_input("T3 (pmol/L)", 0.0, 10.0, 1.5, step=0.1)
        TT4 = st.number_input("TT4 (pmol/L)", 0.1, 300.0, 100.0, step=1.0)
    
    with col2:
        T4U = st.number_input("T4U", 0.0, 10.0, 1.0, step=0.1)
        FTI = st.number_input("FTI", 0.0, 300.0, 100.0, step=1.0)
        TBG = st.number_input("TBG", 0.0, 50.0, 0.0, step=0.5)
    
    with col3:
        st.write("**Measurement Availability**")
        TSH_measured = yes_no("TSH Measured", 1)
        T3_measured = yes_no("T3 Measured", 1)
        TT4_measured = yes_no("TT4 Measured", 1)
        T4U_measured = yes_no("T4U Measured", 1)
        FTI_measured = yes_no("FTI Measured", 1)
        TBG_measured = yes_no("TBG Measured", 0)
    
    # ===== Submit Button =====
    st.subheader("🔮 Make Prediction")
    submit = st.form_submit_button("Predict Thyroid Condition", use_container_width=True)

# =============================================
# Prediction Logic
# =============================================
if submit:
    
    # Calculate ratio features
    TSH_TT4_ratio = TSH / (TT4 + 1e-6)
    T3_TT4_ratio = T3 / (TT4 + 1e-6)
    
    # Apply feature weights (same as training)
    TSH_weighted = TSH * 1.5
    T3_weighted = T3 * 1.25
    TT4_weighted = TT4 * 1.25
    FTI_weighted = FTI * 1.15
    fh_weighted = family_history_level * 1.15
    wc_weighted = weight_change_norm * 1.10
    
    # Create input dataframe
    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "on_thyroxine": on_thyroxine,
        "query_on_thyroxine": query_on_thyroxine,
        "on_antithyroid_medication": on_antithyroid_medication,
        "thyroid_surgery": thyroid_surgery,
        "query_hypothyroid": query_hypothyroid,
        "query_hyperthyroid": query_hyperthyroid,
        "pregnant": pregnant,
        "sick": sick,
        "tumor": tumor,
        "lithium": lithium,
        "goitre": goitre,
        "TSH_measured": TSH_measured,
        "TSH": TSH_weighted,
        "T3_measured": T3_measured,
        "T3": T3_weighted,
        "TT4_measured": TT4_measured,
        "TT4": TT4_weighted,
        "T4U_measured": T4U_measured,
        "T4U": T4U,
        "FTI_measured": FTI_measured,
        "FTI": FTI_weighted,
        "TBG_measured": TBG_measured,
        "TBG": TBG,
        "family_history_level": fh_weighted,
        "weight_change_kg": weight_change_kg,
        "weight_change_norm": wc_weighted,
        "TSH_TT4_ratio": TSH_TT4_ratio,
        "T3_TT4_ratio": T3_TT4_ratio
    }])[FEATURE_COLUMNS]
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    
    # Map prediction to label
    label_map = {0: "Hypothyroid", 1: "Negative"}
    pred_label = label_map[prediction]
    pred_confidence = max(probabilities) * 100
    
    # ===== Display Results =====
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Prediction Result")
        if prediction == 0:  # Hypothyroid
            st.error(f"⚠️ **{pred_label}**")
            st.write(f"""
            The model predicts **hypothyroidism** with **{pred_confidence:.1f}%** confidence.
            
            **Recommendation**: Consult an endocrinologist for:
            - Thyroid function tests
            - TSH level verification
            - Treatment planning if confirmed
            """)
        else:  # Negative
            st.success(f"✅ **{pred_label}**")
            st.write(f"""
            The model predicts **no thyroid disease** with **{pred_confidence:.1f}%** confidence.
            
            **Recommendation**: Continue regular health monitoring and annual checkups.
            """)
    
    with col2:
        st.subheader("📊 Prediction Probabilities")
        prob_data = pd.DataFrame({
            'Condition': ['Hypothyroid', 'Negative'],
            'Probability': [probabilities[0] * 100, probabilities[1] * 100]
        })
        st.bar_chart(prob_data.set_index('Condition'))
        
        st.metric("Hypothyroid Probability", f"{probabilities[0]*100:.2f}%")
        st.metric("Negative Probability", f"{probabilities[1]*100:.2f}%")
    
    # ===== Feature Impact Summary =====
    st.subheader("📋 Input Features Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Patient Profile**")
        st.write(f"Age: {age} years")
        st.write(f"Sex: {'Male' if sex == 1 else 'Female'}")
        st.write(f"Family History Level: {family_history_level}")
    
    with col2:
        st.write("**Weight Change**")
        st.write(f"{weight_change_kg:+.1f} kg")
        st.write(get_weight_change_guidance(weight_change_kg))
    
    with col3:
        st.write("**Hormone Values**")
        st.write(f"TSH: {TSH:.2f} mIU/L")
        st.write(f"T3: {T3:.2f} pmol/L")
        st.write(f"TT4: {TT4:.1f} pmol/L")
        st.write(f"FTI: {FTI:.1f}")
    
    # ===== Input Data Display =====
    with st.expander("📊 Full Feature Vector (for debugging)"):
        st.dataframe(input_data.T)

st.divider()
st.markdown("""
---
**Model Information**: Weighted Voting Ensemble (Random Forest + SVM + KNN)
**Training Data**: UCI Thyroid Dataset with engineered features
**Last Updated**: April 2026

⚠️ **Medical Disclaimer**: This tool is for educational and demonstration purposes only. 
It should not be used for actual medical diagnosis or treatment decisions. 
Always consult qualified healthcare professionals.
""")
