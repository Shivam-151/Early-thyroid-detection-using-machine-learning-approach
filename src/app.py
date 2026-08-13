import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Thyroid Disease Prediction System",
    page_icon="EDT",
    layout="centered"
)

st.title(" Early Detection of Thyroid Disease using Machine Learning")
st.write("Enter patient clinical and hormone values to predict thyroid condition.")


# Load trained model
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "hypothyroid_best_model_correct.pkl"
SCALER_PATH = BASE_DIR / "models" / "hypothyroid_scaler_correct.pkl"

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model()


# Feature order (must match training)
FEATURE_COLUMNS = [
    "age","sex","on_thyroxine","query_on_thyroxine",
    "on_antithyroid_medication","thyroid_surgery",
    "query_hypothyroid","query_hyperthyroid",
    "pregnant","sick","tumor","lithium","goitre",
    "TSH_measured","TSH",
    "T3_measured","T3",
    "TT4_measured","TT4",
    "T4U_measured","T4U",
    "FTI_measured","FTI",
    "TBG_measured","TBG",
    "family_history","weight_change",
    "TSH_TT4_ratio","T3_TT4_ratio"
]


# Helper functions
def yes_no(label, default=0):
    return 1 if st.selectbox(label, ["No","Yes"], index=default) == "Yes" else 0

# User input form
with st.form("prediction_form"):

    st.subheader("Patient Information")

    age = st.number_input("Age", 1, 120, 30)
    sex = 0 if st.selectbox("Sex", ["Female","Male"])=="Female" else 1

    st.subheader("Clinical History")

    on_thyroxine = yes_no("On Thyroxine")
    query_on_thyroxine = yes_no("Query On Thyroxine")
    on_antithyroid_medication = yes_no("Antithyroid Medication")
    thyroid_surgery = yes_no("Thyroid Surgery")
    query_hypothyroid = yes_no("Query Hypothyroid")
    query_hyperthyroid = yes_no("Query Hyperthyroid")
    pregnant = yes_no("Pregnant")
    sick = yes_no("Sick")
    tumor = yes_no("Tumor")
    lithium = yes_no("Lithium")
    goitre = yes_no("Goitre")
    family_history = yes_no("Family History")
    weight_change = yes_no("Weight Change")

    st.subheader("Hormone Values")

    TSH = st.number_input("TSH", 0.0, 100.0, 5.0)
    T3 = st.number_input("T3", 0.0, 10.0, 1.5)
    TT4 = st.number_input("TT4", 0.1, 300.0, 100.0)
    T4U = st.number_input("T4U", 0.0, 10.0, 1.0)
    FTI = st.number_input("FTI", 0.0, 300.0, 100.0)
    TBG = st.number_input("TBG", 0.0, 50.0, 0.0)

    st.subheader("Measurement Availability")

    TSH_measured = yes_no("TSH Measured",1)
    T3_measured = yes_no("T3 Measured",1)
    TT4_measured = yes_no("TT4 Measured",1)
    T4U_measured = yes_no("T4U Measured",1)
    FTI_measured = yes_no("FTI Measured",1)
    TBG_measured = yes_no("TBG Measured",0)

    submit = st.form_submit_button("Predict Thyroid Condition")

# Prediction logic
if submit:

    TSH_TT4_ratio = TSH/(TT4+1e-6)
    T3_TT4_ratio = T3/(TT4+1e-6)

    input_data = pd.DataFrame([{
        "age":age,
        "sex":sex,
        "on_thyroxine":on_thyroxine,
        "query_on_thyroxine":query_on_thyroxine,
        "on_antithyroid_medication":on_antithyroid_medication,
        "thyroid_surgery":thyroid_surgery,
        "query_hypothyroid":query_hypothyroid,
        "query_hyperthyroid":query_hyperthyroid,
        "pregnant":pregnant,
        "sick":sick,
        "tumor":tumor,
        "lithium":lithium,
        "goitre":goitre,
        "TSH_measured":TSH_measured,
        "TSH":TSH*1.5,
        "T3_measured":T3_measured,
        "T3":T3*1.25,
        "TT4_measured":TT4_measured,
        "TT4":TT4*1.25,
        "T4U_measured":T4U_measured,
        "T4U":T4U,
        "FTI_measured":FTI_measured,
        "FTI":FTI*1.15,
        "TBG_measured":TBG_measured,
        "TBG":TBG,
        "family_history":family_history*1.10,
        "weight_change":weight_change*1.10,
        "TSH_TT4_ratio":TSH_TT4_ratio,
        "T3_TT4_ratio":T3_TT4_ratio
    }])[FEATURE_COLUMNS]

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0]

    label_map = {0:"Hypothyroid",1:"Negative"}

    st.subheader("Prediction Result")

    if prediction==0:
        st.error(f" Prediction: {label_map[prediction]}")
    else:
        st.success(f" Prediction: {label_map[prediction]}")

    st.subheader("Prediction Confidence")

    st.write(f"Hypothyroid Probability: {prob[0]*100:.2f}%")
    st.write(f"Negative Probability: {prob[1]*100:.2f}%")

    st.subheader("Processed Feature Vector")

    st.dataframe(input_data)