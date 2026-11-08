"""Train and evaluate the five classifiers required for Assignment 2."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "y"
POSITIVE_LABEL = "yes"
DATA_PATH = Path(__file__).resolve().parent / "Data" / "bank" / "bank-full.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
MODEL_DIR = OUTPUT_DIR / "model"


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Impute, scale, and one-hot encode the input columns."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def get_positive_scores(model: Pipeline, features: pd.DataFrame) -> pd.Series:
    """Return scores oriented toward the positive ('yes') class for AUC."""
    classifier = model.named_steps["classifier"]
    classes = list(classifier.classes_)

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(features)[:, classes.index(POSITIVE_LABEL)]
    else:
        scores = model.decision_function(features)
        if classes.index(POSITIVE_LABEL) == 0:
            scores = -scores
    return pd.Series(scores, index=features.index)


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Place bank-full.csv there."
        )

    data = pd.read_csv(DATA_PATH, sep=";")
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Dataset must contain target column {TARGET_COLUMN!r}.")
    if data[TARGET_COLUMN].isna().any():
        raise ValueError(f"Target column {TARGET_COLUMN!r} contains missing values.")

    features = data.drop(columns=[TARGET_COLUMN])
    target = data[TARGET_COLUMN].astype(str).str.strip().str.lower()
    if set(target.unique()) != {"no", "yes"}:
        raise ValueError(
            f"Expected binary target values 'no' and 'yes'; got {sorted(target.unique())}."
        )
    if features.shape[1] < 12 or len(data) < 500:
        raise ValueError(
            "The dataset must have at least 12 input features and 500 instances."
        )

    numeric_features = features.select_dtypes(include="number").columns.tolist()
    categorical_features = [
        column for column in features.columns if column not in numeric_features
    ]
    if not numeric_features and not categorical_features:
        raise ValueError("No usable input features were found.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    models = {
        "Logistic Regression": (
            "logistic_regression",
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        ),
        "Decision Tree": (
            "decision_tree",
            DecisionTreeClassifier(random_state=RANDOM_STATE),
        ),
        "K-Nearest Neighbors": (
            "knn",
            KNeighborsClassifier(n_neighbors=5),
        ),
        "Gaussian Naive Bayes": (
            "gaussian_nb",
            GaussianNB(),
        ),
        "Support Vector Machine": (
            "support_vector_machine",
            LinearSVC(dual=False, max_iter=5000, random_state=RANDOM_STATE),
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    test_data = x_test.copy()
    test_data[TARGET_COLUMN] = y_test
    test_data.to_csv(OUTPUT_DIR / "test_data.csv", index=False)

    rows = []
    saved_models = {}
    binary_y_test = y_test.eq(POSITIVE_LABEL).astype(int)

    for model_name, (file_name, classifier) in models.items():
        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(numeric_features, categorical_features),
                ),
                ("classifier", classifier),
            ]
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        positive_scores = get_positive_scores(pipeline, x_test)

        rows.append(
            {
                "ML Model Name": model_name,
                "Accuracy": accuracy_score(y_test, predictions),
                "AUC": roc_auc_score(binary_y_test, positive_scores),
                "Precision": precision_score(
                    y_test, predictions, pos_label=POSITIVE_LABEL, zero_division=0
                ),
                "Recall": recall_score(
                    y_test, predictions, pos_label=POSITIVE_LABEL, zero_division=0
                ),
                "F1": f1_score(
                    y_test, predictions, pos_label=POSITIVE_LABEL, zero_division=0
                ),
                "MCC": matthews_corrcoef(y_test, predictions),
            }
        )
        model_path = MODEL_DIR / f"{file_name}.joblib"
        joblib.dump(pipeline, model_path)
        saved_models[model_name] = str(model_path.relative_to(OUTPUT_DIR))
        print(f"Trained and saved {model_name}: {model_path}")

    metrics = pd.DataFrame(rows)
    metrics_path = OUTPUT_DIR / "model_metrics.csv"
    metrics.to_csv(metrics_path, index=False, float_format="%.6f")
    metadata_path = MODEL_DIR / "metadata.json"
    metadata = {
        "target_column": TARGET_COLUMN,
        "positive_label": POSITIVE_LABEL,
        "input_features": features.columns.tolist(),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "models": saved_models,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nEvaluation metrics (positive class: 'yes'):")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Test data saved to: {OUTPUT_DIR / 'test_data.csv'}")
    print(f"Saved models and metadata in: {MODEL_DIR}")


if __name__ == "__main__":
    main()
