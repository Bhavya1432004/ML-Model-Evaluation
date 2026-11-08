# Bank Marketing Classification

## a. Problem statement

Build and compare five classification models that predict whether a bank client
subscribed to a term deposit (`y = yes` or `no`). The same stratified
train/test split is used for every model. Preprocessing is fitted on training
data only, and the saved pipelines are used to evaluate test data without
retraining.

## b. Dataset description — 1 mark

- **Dataset:** Bank Marketing (`bank-full.csv`)
- **Source:** [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Rows:** 45,211
- **Input features:** 16 (numeric and categorical)
- **Target:** `y`, whether the client subscribed to a term deposit (`yes`/`no`)
- **Training/evaluation:** stratified 80/20 split; random state 42
- **Preprocessing:** median imputation and standardization for numeric
  features; most-frequent imputation and one-hot encoding for categorical
  features. Preprocessing is included in each saved model pipeline.
- **Attribution:** S. Moro, P. Cortez, and P. Rita, “A Data-Driven Approach to
  Predict the Success of Bank Telemarketing,” *Decision Support Systems*,
  62, 22–31, 2014. See the UCI page for dataset citation and license details.

The original CSV is semicolon-delimited. `train_models.py` reads it with the
appropriate delimiter. The generated `test_data.csv` is comma-delimited and
contains all 16 features plus the target, as required by the app.

## c. GitHub Repository Link — 1 mark

**Repository:** Replace this placeholder with the public repository URL after
creating it on GitHub:

`https://github.com/Bhavya1432004/ML-Model-Evaluation.git`

The repository should contain the complete project files shown below.

## d. Models used and results — 5 marks

All metrics below were calculated on the same held-out test split. Precision,
recall, and F1 use `yes` as the positive class. AUC is binary ROC AUC using the
model's positive-class probability or decision score.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.901250 | 0.905574 | 0.644483 | 0.347826 | 0.451811 | 0.426058 |
| Decision Tree | 0.874599 | 0.701457 | 0.464880 | 0.475425 | 0.470093 | 0.399021 |
| KNN | 0.896163 | 0.827721 | 0.599002 | 0.340265 | 0.433996 | 0.400128 |
| Naive Bayes (Gaussian) | 0.854805 | 0.810095 | 0.405904 | 0.519849 | 0.455864 | 0.377358 |
| Support Vector Machine | 0.900144 | 0.907201 | 0.660455 | 0.301512 | 0.414017 | 0.401639 |

### Observations — 3 marks

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong overall performance and the highest MCC in this run. Its AUC is also high, though its recall for term-deposit subscriptions is comparatively low. |
| Decision Tree | Finds more positive cases than Logistic Regression and KNN, but has the lowest AUC, suggesting weaker ranking performance on this split. |
| KNN | Accuracy is high, but recall is low; it misses many of the `yes` cases at its default decision rule. |
| Naive Bayes | Has the highest recall, but lower precision and accuracy, so more of its positive predictions are incorrect. |
| Support Vector Machine | Highest AUC and precision in this run, but its recall is the lowest, meaning it identifies fewer actual subscribers. |
| **Overall winner for this dataset?** | **Logistic Regression** is the most balanced choice for this run based on its highest MCC and competitive AUC, precision, and accuracy. If identifying as many subscribers as possible is the priority, Gaussian Naive Bayes has the highest recall. The preferred model depends on the application's objective. |

The dataset is imbalanced, so accuracy alone is not sufficient for choosing a
model. In addition, `duration` is the length of the last contact and may not be
available before a call; interpret results with that potential use-case
limitation in mind.

## Streamlit app

The app implements the assignment's four required interactive features:

| App feature | Marks |
|---|---:|
| CSV test-data upload | 1 |
| Model-selection dropdown | 1 |
| Evaluation metrics | 1 |
| Confusion matrix or classification report | 1 |

It also displays a comparison table for all five models on the uploaded test
data. Upload a labeled test CSV with all required input features and `y`
containing both `yes` and `no`. The app loads the saved model pipelines and
does not retrain them.

### Run locally

From this directory, install the dependencies and start Streamlit:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

For a fresh training run, keep the original dataset at
`Data/bank/bank-full.csv`, then run:

```bash
python train_models.py
```

### Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Sign in to [Streamlit Community Cloud](https://streamlit.io/cloud) with GitHub.
3. Choose **New app**, select this repository and its branch (usually `main`).
4. Set the app file path to `app.py` and click **Deploy**.
5. After deployment, open the live URL and upload `test_data.csv`. Confirm the
   comparison metrics, model dropdown, confusion matrix, and classification
   report all work.

## Project files

```text
bank_marketing_assignment/
├── app.py
├── requirements.txt
├── README.md
├── train_models.py
├── test_data.csv
├── model_metrics.csv
├── Data/
│   └── bank/
│       └── bank-full.csv
└── model/
    ├── decision_tree.joblib
    ├── gaussian_nb.joblib
    ├── knn.joblib
    ├── logistic_regression.joblib
    ├── metadata.json
    └── support_vector_machine.joblib
```
