import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    GridSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def remove_duplicate_records(X, y, disease_name):
    combined = X.copy()
    combined["_target"] = y

    conflicting_targets = (
        combined.groupby(list(X.columns), dropna=False)["_target"]
        .nunique()
        .gt(1)
        .sum()
    )
    if conflicting_targets:
        raise ValueError(
            f"{disease_name} contains duplicate feature rows with conflicting targets."
        )

    before = len(combined)
    combined = combined.drop_duplicates()
    removed = before - len(combined)
    y_clean = combined["_target"]
    X_clean = combined.drop(columns=["_target"])

    if removed:
        print(
            f"Removed {removed} duplicate {disease_name} records "
            f"before splitting."
        )

    return X_clean, y_clean


def train_and_select_best_model(X, y, disease_name):
    X, y = remove_duplicate_records(X, y, disease_name)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    parameter_grids = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=42),
            {"classifier__C": [0.1, 1.0, 10.0]},
        ),
        "SVM": (
            SVC(probability=True, random_state=42),
            {
                "classifier__C": [0.5, 1.0, 10.0],
                "classifier__gamma": ["scale", 0.01],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 10],
                "classifier__min_samples_split": [2, 5],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                eval_metric="logloss",
                random_state=42,
                n_jobs=1,
            ),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [3, 6],
                "classifier__learning_rate": [0.05, 0.1],
            },
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    best_pipeline = None
    best_model_name = ""
    best_cv_roc_auc = -1.0
    best_cv_results = None

    print("\n" + "=" * 60)
    print(f"{disease_name} MODEL RESULTS")
    print("=" * 60)

    best_parameters = None

    for name, (estimator, parameter_grid) in parameter_grids.items():
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", estimator),
            ]
        )
        search = GridSearchCV(
            pipeline,
            parameter_grid,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            refit=True,
        )
        search.fit(X, y)
        tuned_pipeline = search.best_estimator_

        cv_results = cross_validate(
            tuned_pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        mean_roc_auc = cv_results["test_roc_auc"].mean()
        print(
            f"{name:<25} "
            f"CV Accuracy : {cv_results['test_accuracy'].mean():.4f} "
            f"(+/- {cv_results['test_accuracy'].std():.4f})"
        )
        print(
            f"{'':25} "
            f"CV ROC-AUC : {mean_roc_auc:.4f} "
            f"| Precision : {cv_results['test_precision'].mean():.4f} "
            f"| Recall : {cv_results['test_recall'].mean():.4f} "
            f"| F1 : {cv_results['test_f1'].mean():.4f}"
        )
        print(
            f"{'':25} "
            f"Best Parameters : {search.best_params_}"
        )

        if mean_roc_auc > best_cv_roc_auc:
            best_cv_roc_auc = mean_roc_auc
            best_cv_results = cv_results
            best_pipeline = tuned_pipeline
            best_model_name = name
            best_parameters = search.best_params_

    best_pipeline.fit(X_train, y_train)
    best_predictions = best_pipeline.predict(X_test)

    print("\nBest Model :", best_model_name)
    print("Best Mean CV ROC-AUC :", round(best_cv_roc_auc, 4))
    print("Best Parameters :", best_parameters)
    print(
        "Mean CV Accuracy :",
        round(best_cv_results["test_accuracy"].mean(), 4),
        "+/-",
        round(best_cv_results["test_accuracy"].std(), 4),
    )

    held_out_accuracy = accuracy_score(y_test, best_predictions)
    print("\nHeld-Out Accuracy :", round(held_out_accuracy, 4))
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, best_predictions))
    print("\nClassification Report:\n")
    print(classification_report(y_test, best_predictions))

    if hasattr(best_pipeline, "predict_proba"):
        positive_probabilities = best_pipeline.predict_proba(X_test)[:, 1]
        held_out_roc_auc = roc_auc_score(y_test, positive_probabilities)
        print("Held-Out ROC-AUC :", round(held_out_roc_auc, 4))

    return (
        best_pipeline,
        best_model_name,
        held_out_accuracy,
        {
            "cv_accuracy_mean": float(best_cv_results["test_accuracy"].mean()),
            "cv_accuracy_std": float(best_cv_results["test_accuracy"].std()),
            "cv_precision": float(best_cv_results["test_precision"].mean()),
            "cv_recall": float(best_cv_results["test_recall"].mean()),
            "cv_f1": float(best_cv_results["test_f1"].mean()),
            "cv_roc_auc": float(best_cv_roc_auc),
            "held_out_accuracy": float(held_out_accuracy),
            "best_parameters": best_parameters,
        },
    )


def save_artifact(disease, pipeline, model_name, accuracy, metrics, feature_names,
                  positive_label):
    artifact = {
        "pipeline": pipeline,
        "model_name": model_name,
        "accuracy": accuracy,
        "metrics": metrics,
        "feature_names": list(feature_names),
        "positive_label": positive_label,
    }
    joblib.dump(artifact, MODELS_DIR / f"{disease}_artifact.pkl")
    print(f"{disease.replace('_', ' ').title()} artifact saved")


def train_diabetes():
    columns = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
        "Outcome",
    ]
    diabetes_df = pd.read_csv(DATASETS_DIR / "diabetes.csv", names=columns)
    for column in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        diabetes_df[column] = diabetes_df[column].replace(0, np.nan)
    X = diabetes_df.drop(columns=["Outcome"])
    y = diabetes_df["Outcome"]

    pipeline, model_name, accuracy, metrics = train_and_select_best_model(
        X, y, "DIABETES"
    )
    save_artifact(
        "diabetes",
        pipeline,
        model_name,
        accuracy,
        metrics,
        X.columns,
        "High risk of diabetes",
    )


def train_heart():
    heart_df = pd.read_csv(DATASETS_DIR / "heart.csv")
    X = heart_df.drop(columns=["target"])
    y = heart_df["target"]

    pipeline, model_name, accuracy, metrics = train_and_select_best_model(
        X, y, "HEART DISEASE"
    )
    save_artifact(
        "heart",
        pipeline,
        model_name,
        accuracy,
        metrics,
        X.columns,
        "Heart disease detected",
    )


def train_breast_cancer():
    breast_df = pd.read_csv(DATASETS_DIR / "breast_cancer.csv")
    breast_df = breast_df.drop(
        columns=["id"]
        if "id" in breast_df.columns
        else []
    )
    breast_df = breast_df.drop(
        columns=[
            column
            for column in breast_df.columns
            if "Unnamed" in column
        ]
    )

    encoder = LabelEncoder()
    breast_df["diagnosis"] = encoder.fit_transform(breast_df["diagnosis"])
    X = breast_df.drop(columns=["diagnosis"])
    y = breast_df["diagnosis"]

    pipeline, model_name, accuracy, metrics = train_and_select_best_model(
        X, y, "BREAST CANCER"
    )
    save_artifact(
        "breast_cancer",
        pipeline,
        model_name,
        accuracy,
        metrics,
        X.columns,
        "Malignant cancer detected",
    )


def main():
    print("\nLoading Diabetes Dataset...")
    train_diabetes()
    print("\nLoading Heart Disease Dataset...")
    train_heart()
    print("\nLoading Breast Cancer Dataset...")
    train_breast_cancer()

    print("\n" + "=" * 60)
    print("ALL MODELS TRAINED SUCCESSFULLY")
    print("ARTIFACTS SAVED IN models FOLDER")
    print("=" * 60)


if __name__ == "__main__":
    main()