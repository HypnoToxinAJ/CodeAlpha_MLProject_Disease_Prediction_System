import unittest
from pathlib import Path

import joblib
import pandas as pd
import math

from disease_prediction_system import remove_duplicate_records


PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"


class PredictionArtifactTests(unittest.TestCase):
    def test_diabetes_pipeline_handles_missing_values(self):
        artifact = joblib.load(MODELS_DIR / "diabetes_artifact.pkl")
        sample = pd.DataFrame(
            [[1, None, 70, None, None, None, 0.4, 33]],
            columns=artifact["feature_names"],
        )
        prediction = artifact["pipeline"].predict(sample)
        self.assertIn(prediction[0], (0, 1))

    def test_prediction_probability_is_available(self):
        artifact = joblib.load(MODELS_DIR / "diabetes_artifact.pkl")
        sample = pd.DataFrame(
            [[1, 120, 70, 20, 79, 25.0, 0.4, 33]],
            columns=artifact["feature_names"],
        )
        probability = artifact["pipeline"].predict_proba(sample)[0, 1]
        self.assertTrue(math.isfinite(probability))
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)

    def test_duplicate_records_are_removed_consistently(self):
        features = pd.DataFrame(
            {"age": [50, 50, 60], "cholesterol": [200, 200, 220]}
        )
        targets = pd.Series([0, 0, 1])

        clean_features, clean_targets = remove_duplicate_records(
            features, targets, "test"
        )

        self.assertEqual(len(clean_features), 2)
        self.assertEqual(len(clean_targets), 2)

    def test_diabetes_artifact_has_expected_features(self):
        artifact = joblib.load(MODELS_DIR / "diabetes_artifact.pkl")
        self.assertEqual(len(artifact["feature_names"]), 8)
        self.assertIn("cv_roc_auc", artifact["metrics"])
        self.assertIn("held_out_accuracy", artifact["metrics"])
        self.assertIn("best_parameters", artifact["metrics"])

        sample = pd.DataFrame(
            [[1, 120, 70, 20, 79, 25.0, 0.4, 33]],
            columns=artifact["feature_names"],
        )
        prediction = artifact["pipeline"].predict(sample)
        self.assertIn(prediction[0], (0, 1))

    def test_heart_artifact_has_expected_features(self):
        artifact = joblib.load(MODELS_DIR / "heart_artifact.pkl")
        self.assertEqual(len(artifact["feature_names"]), 13)

        sample = pd.DataFrame(
            [[52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3]],
            columns=artifact["feature_names"],
        )
        prediction = artifact["pipeline"].predict(sample)
        self.assertIn(prediction[0], (0, 1))

    def test_breast_cancer_artifact_has_expected_features(self):
        artifact = joblib.load(MODELS_DIR / "breast_cancer_artifact.pkl")
        self.assertEqual(len(artifact["feature_names"]), 30)

        values = [[15.0] * len(artifact["feature_names"])]
        sample = pd.DataFrame(values, columns=artifact["feature_names"])
        prediction = artifact["pipeline"].predict(sample)
        self.assertIn(prediction[0], (0, 1))


if __name__ == "__main__":
    unittest.main()
