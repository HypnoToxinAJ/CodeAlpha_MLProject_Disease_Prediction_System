from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


def load_artifact(disease):
    artifact_path = MODELS_DIR / f"{disease}_artifact.pkl"
    if not artifact_path.exists():
        st.error(
            f"The {disease.replace('_', ' ')} model is missing. "
            "Run `python disease_prediction_system.py` once to train and save artifacts."
        )
        st.stop()
    try:
        return joblib.load(artifact_path)
    except Exception as exc:
        st.error(f"Could not load the {disease.replace('_', ' ')} model: {exc}")
        st.stop()


def predict(disease, values):
    artifact = load_artifact(disease)
    data = pd.DataFrame([values], columns=artifact["feature_names"])
    prediction = artifact["pipeline"].predict(data)[0]
    probability = artifact["pipeline"].predict_proba(data)[0, 1]

    result = (
        artifact["positive_label"]
        if prediction == 1
        else "No positive risk detected"
    )
    st.metric("Prediction", result)
    st.progress(float(probability), text=f"Positive probability: {probability:.1%}")


st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🏥",
)
st.title("Disease Prediction System")
st.warning(
    "Educational use only. This application is not a medical diagnosis and "
    "does not replace professional healthcare advice."
)

disease = st.sidebar.selectbox(
    "Choose a prediction",
    ["Diabetes", "Heart Disease", "Breast Cancer"],
)

if disease == "Diabetes":
    st.subheader("Diabetes details")
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose (0 if unknown)", 0, 300, 120)
    blood_pressure = st.number_input("Blood Pressure (0 if unknown)", 0, 200, 70)
    skin_thickness = st.number_input("Skin Thickness (0 if unknown)", 0, 100, 20)
    insulin = st.number_input("Insulin (0 if unknown)", 0, 900, 79)
    bmi = st.number_input("BMI (0 if unknown)", 0.0, 100.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 10.0, 0.4)
    age = st.number_input("Age", 1, 120, 33)

    values = [
        pregnancies,
        glucose or None,
        blood_pressure or None,
        skin_thickness or None,
        insulin or None,
        bmi or None,
        dpf,
        age,
    ]

elif disease == "Heart Disease":
    st.subheader("Heart disease details")
    values = [
        st.number_input("Age", 1, 120, 52),
        st.number_input("Sex (0=Female, 1=Male)", 0, 1, 1),
        st.number_input("Chest Pain Type", 0, 3, 0),
        st.number_input("Resting Blood Pressure", 0, 300, 125),
        st.number_input("Cholesterol", 0, 700, 212),
        st.number_input("Fasting Blood Sugar", 0, 1, 0),
        st.number_input("Rest ECG", 0, 2, 1),
        st.number_input("Maximum Heart Rate", 0, 250, 168),
        st.number_input("Exercise Angina", 0, 1, 0),
        st.number_input("Oldpeak", -2.0, 10.0, 1.0),
        st.number_input("Slope", 0, 2, 2),
        st.number_input("CA", 0, 4, 2),
        st.number_input("Thal", 0, 3, 3),
    ]

else:
    st.subheader("Breast cancer measurements")
    artifact = load_artifact("breast_cancer")
    values = [
        st.number_input(feature, 0.0, 1000.0, 15.0)
        for feature in artifact["feature_names"]
    ]

if st.button("Predict", type="primary"):
    disease_key = {
        "Diabetes": "diabetes",
        "Heart Disease": "heart",
        "Breast Cancer": "breast_cancer",
    }[disease]
    predict(disease_key, values)
