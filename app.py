"""Streamlit interface for evaluating the saved Bank Marketing classifiers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
METADATA_PATH = ROOT / "model" / "metadata.json"
EXPECTED_LABELS = {"no", "yes"}

st.set_page_config(page_title="Bank Marketing Model Evaluation", page_icon="📊")
st.title("Bank Marketing Classifier Evaluation")
st.write(
    "Upload labeled test data to compare the five saved classifiers. "
    "The app loads trained pipelines and never retrains on uploaded data."
)
st.info(
    "Upload test data only. The CSV must contain the 16 input features and "
    "the target column `y` with `yes`/`no` labels."
)


@st.cache_data
def load_metadata(metadata_path: str) -> dict:
    with Path(metadata_path).open(encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


if not METADATA_PATH.is_file():
    st.error(f"Model metadata was not found: {METADATA_PATH}")
    st.stop()

metadata = load_metadata(str(METADATA_PATH))
target_column = metadata["target_column"]
positive_label = metadata["positive_label"]
input_features = metadata["input_features"]
model_paths = metadata["models"]

uploaded_file = st.file_uploader(
    "Upload labeled test data (CSV)", type=["csv"], max_upload_size=20
)
if uploaded_file is None:
    st.caption("The repository includes `test_data.csv` to try the app.")
    st.stop()

try:
    uploaded_data = pd.read_csv(uploaded_file, sep=None, engine="python")
except (UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
    st.error(f"Could not read the uploaded CSV: {error}")
    st.stop()

missing_columns = [
    column
    for column in [*input_features, target_column]
    if column not in uploaded_data.columns
]
if missing_columns:
    st.error(f"The CSV is missing required columns: {', '.join(missing_columns)}")
    st.stop()

if uploaded_data.empty:
    st.error("The uploaded CSV contains no data rows.")
    st.stop()

if uploaded_data[target_column].isna().any():
    st.error(f"Target column `{target_column}` contains missing values.")
    st.stop()

y_true = uploaded_data[target_column].astype(str).str.strip().str.lower()
if not set(y_true.unique()).issubset(EXPECTED_LABELS):
    st.error("Target column `y` must contain only `yes` and `no` values.")
    st.stop()
if set(y_true.unique()) != EXPECTED_LABELS:
    st.error("The test data must contain at least one `yes` and one `no` row.")
    st.stop()

x_test = uploaded_data.loc[:, input_features]
binary_y_true = y_true.eq(positive_label).astype(int)
results: dict[str, dict] = {}

for model_name, relative_model_path in model_paths.items():
    model_path = ROOT / relative_model_path
    if not model_path.is_file():
        st.error(f"Saved model for {model_name} was not found: {model_path}")
        st.stop()

    model = load_model(str(model_path))
    predictions = pd.Series(model.predict(x_test), index=x_test.index).astype(str)
    classifier = model.named_steps["classifier"]
    classes = list(classifier.classes_)

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(x_test)[:, classes.index(positive_label)]
    else:
        scores = model.decision_function(x_test)
        if classes.index(positive_label) == 0:
            scores = -scores

    results[model_name] = {
        "predictions": predictions,
        "scores": scores,
        "Accuracy": accuracy_score(y_true, predictions),
        "AUC": roc_auc_score(binary_y_true, scores),
        "Precision": precision_score(
            y_true, predictions, pos_label=positive_label, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, pos_label=positive_label, zero_division=0
        ),
        "F1": f1_score(
            y_true, predictions, pos_label=positive_label, zero_division=0
        ),
        "MCC": matthews_corrcoef(y_true, predictions),
    }

st.subheader("Model comparison on uploaded test data")
comparison = pd.DataFrame(
    [
        {
            "ML Model Name": name,
            **{
                metric: result[metric]
                for metric in ("Accuracy", "AUC", "Precision", "Recall", "F1", "MCC")
            },
        }
        for name, result in results.items()
    ]
)
st.dataframe(
    comparison.style.format(
        {
            "Accuracy": "{:.4f}",
            "AUC": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1": "{:.4f}",
            "MCC": "{:.4f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

selected_model_name = st.selectbox("Inspect a model", list(results))
selected_result = results[selected_model_name]
st.subheader(f"{selected_model_name}: evaluation metrics")
metric_columns = st.columns(6)
for column, metric in zip(
    metric_columns, ("Accuracy", "AUC", "Precision", "Recall", "F1", "MCC")
):
    column.metric(metric, f"{selected_result[metric]:.4f}")

st.subheader("Confusion matrix")
matrix = confusion_matrix(y_true, selected_result["predictions"], labels=["no", "yes"])
st.dataframe(
    pd.DataFrame(
        matrix,
        index=["Actual no", "Actual yes"],
        columns=["Predicted no", "Predicted yes"],
    ),
    use_container_width=True,
)

st.subheader("Classification report")
report = classification_report(
    y_true,
    selected_result["predictions"],
    labels=["no", "yes"],
    target_names=["no", "yes"],
    output_dict=True,
    zero_division=0,
)
st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)
