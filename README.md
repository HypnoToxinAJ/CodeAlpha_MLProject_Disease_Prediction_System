# Disease Prediction System

A Python machine-learning experiment that compares classifiers for diabetes,
heart disease, and breast cancer datasets. It includes a command-line predictor
and a Streamlit web interface. This GitHub repository contains source code and
documentation only; health-related CSV files and model artifacts are excluded.

> **Important:** This is an educational experiment, not a medical device. Its
> predictions and probabilities are not diagnoses, medical advice, or a
> substitute for a qualified healthcare professional.

## What the experiment does

For each dataset, the training script:

1. Loads the local CSV and separates its input features from the target.
2. Removes duplicate feature/target rows. If identical features have
   conflicting targets, training stops with an error.
3. Creates a stratified 80/20 train/test split using random seed 42.
4. Builds pipelines that impute missing numeric values with the median, scale
   features, and fit a classifier.
5. Tunes a small parameter grid for Logistic Regression, Support Vector
   Machine (SVM), Random Forest, and XGBoost using stratified 5-fold
   cross-validation and ROC-AUC as the search score.
6. Prints cross-validation metrics and a test-split confusion matrix,
   classification report, accuracy, and ROC-AUC.
7. Saves the selected pipeline and metadata in `models/`.

The diabetes data uses zero as a missing-value marker for glucose, blood
pressure, skin thickness, insulin, and BMI. Those zeros are converted to
missing values before fitting the pipeline. Other zero values are retained.

## Current observed results

These are results from the current saved experiment run:

| Dataset | Selected model | Mean 5-fold CV accuracy | Test-split accuracy | Test-split ROC-AUC |
|---|---|---:|---:|---:|
| Diabetes | SVM | 76.3% | 69.5% | 81.5% |
| Heart disease | Random Forest | 82.5% | 77.0% | 85.7% |
| Breast cancer | Logistic Regression | 97.4% | 98.2% | 99.8% |

These values are dataset- and split-specific, not guaranteed performance.
Accuracy alone does not describe the cost of false negatives or false
positives; review recall, precision, F1-score, and the confusion matrix in the
training output as well.

### Evaluation caveat

The current implementation performs hyperparameter search on the full dataset
before reporting held-out test results. As a result, the test split influenced
model and parameter selection, and these test scores are not an independent
final evaluation. The reported cross-validation results also follow parameter
selection on the full dataset, so they may be optimistic.

Treat all scores above as exploratory. Before making performance claims, update
the evaluation to split off the test set first, tune and select models using
training data only (ideally with nested cross-validation), then evaluate the
chosen model once on the untouched test set. Any medical application would also
require suitable external validation, clinical review, privacy controls, and
regulatory assessment.

### Duplicate records in the heart dataset

The bundled heart CSV contains 1,025 rows, of which 723 are duplicate
feature/target rows. The training script removes those duplicates before
splitting, leaving 302 unique rows for this experiment. This substantially
reduces the effective dataset size and is one reason the unique-record
evaluation differs from the original row-count result. The repository does
not establish the provenance or collection process for this CSV; verify its
source and usage terms before redistribution or research use.

## Features

- Predicts diabetes, heart-disease, and breast-cancer dataset labels.
- Compares Logistic Regression, SVM, Random Forest, and XGBoost.
- Tunes each classifier using a defined grid and stratified folds.
- Reports accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix,
  and classification report.
- Persists trained preprocessing-and-classifier pipelines with Joblib.
- Supports numeric command-line input and a Streamlit interface.
- Displays positive-class model probability as an experimental score.
- Validates command-line inputs and handles missing diabetes measurements.
- Includes basic artifact and prediction tests.

Probabilities are model outputs and have not been calibrated. They should not
be interpreted as a person's clinical probability of disease.

## Data

The training script expects local CSV files in `datasets/`. They are not
included in this public repository because they contain health-related records.
Obtain each dataset only from a source whose terms permit your intended use,
and place the authorized file in the expected path before training. Do not
commit datasets, patient records, or trained artifacts to this repository.

### Diabetes (`diabetes.csv`)

Eight input columns: pregnancies, glucose, blood pressure, skin thickness,
insulin, BMI, diabetes pedigree function, and age. The final column is the
binary `Outcome` target.

### Heart disease (`heart.csv`)

Thirteen input columns: age, sex, chest-pain type, resting blood pressure,
cholesterol, fasting blood sugar, resting ECG, maximum heart rate, exercise
angina, oldpeak, slope, CA, and thal. The `target` column is the binary label.

### Breast cancer (`breast_cancer.csv`)

Thirty numeric measurements derived from cell nuclei, including mean,
standard-error, and worst measurements for radius, texture, perimeter, area,
smoothness, compactness, concavity, concave points, symmetry, and fractal
dimension. The `diagnosis` column is encoded as a binary target; the `id`
column, when present, is excluded from training.

The dataset descriptions follow the existing project documentation. Consult
the original dataset sources to confirm provenance, licensing, definitions,
and appropriate use. The project does not bundle data-download scripts or
guarantee that a particular copy is suitable for research or redistribution.

## Project layout

```text
Disease_Prediction_System/
├── .devcontainer/
│   └── devcontainer.json
├── .streamlit/
│   └── config.toml
├── datasets/
│   └── .gitkeep                  # Put authorized local CSV files here
├── models/
│   └──                           # Artifacts generated locally; not committed
├── app.py
├── disease_prediction_system.py
├── prediction_menu.py
├── requirements.txt
└── test_prediction_artifacts.py
```

The CSV files and generated `*_artifact.pkl` files are intentionally not
committed. After cloning, add authorized data files to `datasets/`, train the
models, then run either prediction interface. The training script creates the
required artifacts locally under `models/`.

## Requirements and installation

Python 3.10 or later is recommended. From the project directory:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the experiment

First place the authorized CSVs at these paths:

```text
datasets/diabetes.csv
datasets/heart.csv
datasets/breast_cancer.csv
```

Train models, evaluate them, and create the artifacts:

```bash
python disease_prediction_system.py
```

Run the interactive terminal menu:

```bash
python prediction_menu.py
```

Run the web interface:

```bash
python -m streamlit run app.py
```

Then open the local URL printed by Streamlit (usually
`http://localhost:8501`). If a port is already in use, choose another, for
example:

```bash
python -m streamlit run app.py --server.port 8511
```

The web app allows selection of one of the three tasks, entry of feature
values, and display of the predicted label and positive-class model score.
For unknown diabetes measurements, use zero; the app treats zero as missing
for the five measurements listed above.

## Launch in GitHub Codespaces

The `.devcontainer/devcontainer.json` configuration installs the dependencies,
starts Streamlit, and forwards port 8501.

1. Push this project to a GitHub repository.
2. Open the repository and select **Code → Codespaces → Create codespace**.
3. Wait for the Codespace setup to finish.
4. Open the forwarded port 8501 from the **Ports** panel.

Before predictions will work in a fresh Codespace, add the authorized CSV files
to `datasets/` and run `python disease_prediction_system.py` in its terminal.
Refresh the app after training. The CSV and model artifacts are ignored by Git
and are not uploaded. GitHub Pages is static hosting and cannot run this
Python/Streamlit server.

## Tests

From the project directory, with dependencies installed:

```bash
python -m unittest -v test_prediction_artifacts.py
```

The tests check that saved artifacts load, accept correctly shaped inputs,
handle missing diabetes features, return binary predictions, expose a valid
probability, and remove duplicate rows.

## Dependencies

The packages listed in `requirements.txt` are:

- pandas and NumPy for data handling
- scikit-learn for preprocessing, pipelines, models, validation, and metrics
- XGBoost for gradient-boosted trees
- Joblib for saving/loading model artifacts
- Streamlit for the web application

## Responsible-use note

Do not use this experiment to make healthcare decisions or to diagnose,
screen, treat, or rule out disease. The supplied datasets may be small,
duplicated, biased, or unrepresentative. No patient privacy, clinical
validation, calibration, fairness, or regulatory suitability is established
by this project.
