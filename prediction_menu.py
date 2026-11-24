import math
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


def load_artifact(disease):
    try:
        return joblib.load(MODELS_DIR / f"{disease}_artifact.pkl")
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Missing {disease} model artifact. Run disease_prediction_system.py first."
        ) from exc


def prompt_number(prompt, minimum=0.0, maximum=None):
    while True:
        try:
            value = float(input(prompt))
        except ValueError:
            print("Please enter a valid number.")
            continue

        if not math.isfinite(value):
            print("Please enter a finite number.")
            continue
        if value < minimum:
            print(f"Value must be at least {minimum}.")
            continue
        if maximum is not None and value > maximum:
            print(f"Value must be no more than {maximum}.")
            continue
        return value


def predict(artifact, values):
    data = pd.DataFrame([values], columns=artifact["feature_names"])
    prediction = artifact["pipeline"].predict(data)[0]
    probability = None
    if hasattr(artifact["pipeline"], "predict_proba"):
        probability = artifact["pipeline"].predict_proba(data)[0, 1]

    if prediction == 1:
        print(f"\n{artifact['positive_label'].upper()}")
    else:
        print("\nNO POSITIVE RISK DETECTED")

    if probability is not None:
        print(f"Positive prediction probability: {probability:.1%}")


def diabetes_prediction():
    artifact = load_artifact("diabetes")
    print("\nENTER DIABETES DETAILS\n")
    values = [
        prompt_number("Pregnancies: ", maximum=20),
        prompt_number("Glucose (0 if unknown): ", maximum=300),
        prompt_number("Blood Pressure (0 if unknown): ", maximum=200),
        prompt_number("Skin Thickness (0 if unknown): ", maximum=100),
        prompt_number("Insulin (0 if unknown): ", maximum=900),
        prompt_number("BMI (0 if unknown): ", maximum=100),
        prompt_number("Diabetes Pedigree Function: ", maximum=10),
        prompt_number("Age: ", minimum=1, maximum=120),
    ]
    for index in (1, 2, 3, 4, 5):
        if values[index] == 0:
            values[index] = None
    predict(artifact, values)


def heart_prediction():
    artifact = load_artifact("heart")
    print("\nENTER HEART DISEASE DETAILS\n")
    values = [
        prompt_number("Age: ", minimum=1, maximum=120),
        prompt_number("Sex (0=Female,1=Male): ", maximum=1),
        prompt_number("Chest Pain Type (0-3): ", maximum=3),
        prompt_number("Resting Blood Pressure: ", maximum=300),
        prompt_number("Cholesterol: ", maximum=700),
        prompt_number("Fasting Blood Sugar (0 or 1): ", maximum=1),
        prompt_number("Rest ECG (0-2): ", maximum=2),
        prompt_number("Max Heart Rate: ", maximum=250),
        prompt_number("Exercise Angina (0 or 1): ", maximum=1),
        prompt_number("Oldpeak: ", minimum=-2, maximum=10),
        prompt_number("Slope (0-2): ", maximum=2),
        prompt_number("CA (0-4): ", maximum=4),
        prompt_number("Thal (0-3): ", maximum=3),
    ]
    predict(artifact, values)


def breast_prediction():
    artifact = load_artifact("breast_cancer")
    print("\nBREAST CANCER PREDICTION")
    print("\nEnter 30 Features")
    values = [
        prompt_number(f"{feature}: ", maximum=1000)
        for feature in artifact["feature_names"]
    ]
    predict(artifact, values)


def main():
    print(
        "\nMedical disclaimer: This project is for education and experimentation "
        "only. Its predictions are not medical diagnoses or professional advice."
    )
    while True:
        print("\n" + "=" * 50)
        print("DISEASE PREDICTION SYSTEM")
        print("=" * 50)
        print("1. Diabetes Prediction")
        print("2. Heart Disease Prediction")
        print("3. Breast Cancer Prediction")
        print("4. Exit")

        choice = input("\nEnter Choice: ")
        if choice == "1":
            diabetes_prediction()
        elif choice == "2":
            heart_prediction()
        elif choice == "3":
            breast_prediction()
        elif choice == "4":
            print("\nThank You")
            break
        else:
            print("\nInvalid Choice")


if __name__ == "__main__":
    main()