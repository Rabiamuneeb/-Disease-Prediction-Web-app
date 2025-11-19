# ============================================================
# 🩺 Disease Prediction App
# Datasets: Breast Cancer, Diabetes, Heart Disease
# Algorithms: Logistic Regression, SVM, Random Forest, XGBoost
# ============================================================

import pandas as pd
import numpy as np
import streamlit as st

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

# XGBoost import
from xgboost import XGBClassifier

# ------------------------------------------------------------
# Streamlit page setup
# ------------------------------------------------------------
st.set_page_config(page_title="Disease Prediction", page_icon="🩺", layout="centered")

# ------------------------------------------------------------
# File paths (make sure these CSVs are in the same folder)
# ------------------------------------------------------------
PATHS = {
    "breast_cancer": "data.csv",
    "diabetes": "diabetes.csv",
    "cardio": "cardio.csv"
}

# Load datasets
DFS = {}
for name, path in PATHS.items():
    try:
        DFS[name] = pd.read_csv(path)
    except Exception:
        DFS[name] = None

# ------------------------------------------------------------
# Function: Prepare schema for each dataset
# ------------------------------------------------------------
def prepare_schema(df, dataset_name):
    """Return features (X), target (y), and column types for preprocessing."""

    if dataset_name == "breast_cancer":
        # Target column
        target = "diagnosis"
        df[target] = df[target].map({"B": 0, "M": 1})
        df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")

        # Auto-select all '_mean' features
        feature_cols = [c for c in df.columns if "_mean" in c]
        X = df[feature_cols].copy()
        y = df[target].copy()

    elif dataset_name == "diabetes":
        target = "Outcome"
        feature_cols = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]
        X = df[feature_cols].copy()
        y = df[target].copy()

    elif dataset_name == "cardio":
        target = "num"
        y = (df[target] > 0).astype(int)

        # Use common UCI heart features
        req = ["age", "sex", "cp", "trestbps", "chol", "fbs",
               "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
        feature_cols = [c for c in req if c in df.columns]
        X = df[feature_cols].copy()

    else:
        raise ValueError("Unknown dataset")

    # Identify numeric and categorical columns
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    return X, y, num_cols, cat_cols

# ------------------------------------------------------------
# Function: Preprocessing pipeline
# ------------------------------------------------------------
def make_preprocess(num_cols, cat_cols):
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols)
    ])

# ------------------------------------------------------------
# Function: Train multiple models
# ------------------------------------------------------------
def train_models(X, y, preprocess):
    """Train Logistic Regression, SVM, Random Forest, XGBoost and return results."""

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    }

    results = {}
    trained_models = {}

    # Train and evaluate each model
    for name, clf in models.items():
        pipe = Pipeline([
            ("prep", preprocess),
            ("clf", clf)
        ])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, proba)
        ap = average_precision_score(y_test, proba)
        results[name] = (auroc, ap)
        trained_models[name] = pipe

    return results, trained_models

# ------------------------------------------------------------
# UI: Title
# ------------------------------------------------------------
st.title("🩺 Disease Prediction (Breast Cancer, Diabetes, Cardio)")
st.caption("Enter lab and report values to estimate disease risk with a clear probability. This is an educational tool, not a medical device.")

# ------------------------------------------------------------
# Show performance for each dataset
# ------------------------------------------------------------
for dataset_name in ["breast_cancer", "diabetes", "cardio"]:
    df = DFS.get(dataset_name)
    st.subheader(f"Model performance — {dataset_name}")
    if df is None:
        st.warning(f"File not found: {PATHS[dataset_name]}")
        continue
    try:
        X, y, num_cols, cat_cols = prepare_schema(df, dataset_name)
        preprocess = make_preprocess(num_cols, cat_cols)
        results, _ = train_models(X, y, preprocess)

        # Display results in table
        perf_table = pd.DataFrame(results, index=["AUROC", "AP"]).T
        st.dataframe(perf_table.style.format("{:.3f}"))
    except Exception as e:
        st.error(f"Setup error for {dataset_name}: {e}")

st.divider()

# ------------------------------------------------------------
# Interactive prediction form
# ------------------------------------------------------------
st.header("🔮 Patient-friendly prediction with probability")

dataset_choice = st.selectbox("Choose the disease area", ["breast_cancer", "diabetes", "cardio"], index=1)
df_choice = DFS.get(dataset_choice)
if df_choice is None:
    st.error(f"Cannot load {dataset_choice}. Please ensure the file exists: {PATHS[dataset_choice]}")
    st.stop()

try:
    Xc, yc, num_cols_c, cat_cols_c = prepare_schema(df_choice, dataset_choice)
    preprocess_c = make_preprocess(num_cols_c, cat_cols_c)
    results_c, trained_models_c = train_models(Xc, yc, preprocess_c)
except Exception as e:
    st.error(f"Setup error: {e}")
    st.stop()

# Use Random Forest for prediction form
model_c = trained_models_c["Random Forest"]

st.write("Enter patient/lab values below. All fields are required to ensure accurate prediction.")

user_inputs = {}
feature_columns = Xc.columns.tolist()

with st.form("prediction_form"):
    for col in feature_columns:
        series = Xc[col]
        if series.dtype.kind in "biu":  # integer
            val = int(series.mean()) if pd.to_numeric(series, errors="coerce").notna().any() else 0
            user_inputs[col] = st.number_input(col, value=val, step=1)
        elif series.dtype.kind == "f":  # float
            val = float(series.mean()) if pd.to_numeric(series, errors="coerce").notna().any() else 0.0
            user_inputs[col] = st.number_input(col, value=float(f"{val:.3f}"))
        else:
            choices = [str(x) for x in series.dropna().unique().tolist()]
            if not choices:
                choices = ["unknown"]
            user_inputs[col] = st.selectbox(col, choices)

    submitted = st.form_submit_button("Predict disease risk")

if submitted:
    input_df = pd.DataFrame([user_inputs])[feature_columns]
    pred = model_c.predict(input_df)[0]
    prob = model_c.predict_proba(input_df)[0][1]
    percent = prob * 100.0

    if pred == 1:
        st.error(f"⚠️ High risk of disease — estimated probability: {percent:.1f}%")
    else:
        st.success(f"✅ Low risk of disease — estimated probability: {percent:.1f}%")

    st.info(f"Model quality for {dataset_choice} (Random Forest): "
            f"AUROC={results_c['Random Forest'][0]:.3f}, AP={results_c['Random Forest'][1]:.3f}")