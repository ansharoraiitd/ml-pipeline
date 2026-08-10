"""
Day 3 — Build the End-to-End ML Pipeline

WHAT THIS DOES:
Takes the cleaned Adult Census Income data and builds a proper
scikit-learn Pipeline + ColumnTransformer so preprocessing and
modeling happen together.

The key lesson:
- preprocessing is fit ONLY on training data
- test data is transformed using the already-fitted preprocessing
- no preprocessing leakage
- the entire workflow can be saved and reused as one object
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

df = pd.read_csv("adult_income_raw.csv")

print(f"Raw shape: {df.shape}")


# ============================================================
# 2. CLEAN TARGET + MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("CLEANING DATA")
print("=" * 60)

# Drop redundant education column.
# education-num contains the same underlying information in
# numerical form, so keeping both would duplicate the signal.
df = df.drop(columns=["education"])

# Drop fnlwgt because it is a census sampling weight rather
# than a meaningful individual-level predictive feature.
df = df.drop(columns=["fnlwgt"])

# Replace disguised missing values with actual NaN.
categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns

for col in categorical_columns:
    df[col] = df[col].replace("?", np.nan)

# Convert target into binary:
# <=50K  -> 0
# >50K   -> 1
df["income"] = (
    df["income"]
    .astype(str)
    .str.strip()
    .str.replace(".", "", regex=False)
    .map({
        "<=50K": 0,
        ">50K": 1
    })
)

# Separate features and target.
X = df.drop(columns=["income"])
y = df["income"]

print(f"Cleaned shape: {df.shape}")
print(f"Remaining missing values: {df.isnull().sum().sum()}")
print(f"Positive class rate: {y.mean():.3f}")


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training rows: {len(X_train)}")
print(f"Test rows:     {len(X_test)}")


# ============================================================
# 4. IDENTIFY COLUMN TYPES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nNumerical features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 5. NUMERICAL PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ============================================================
# 6. CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# ============================================================
# 7. COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_pipeline,
            numeric_features
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 8. BUILD COMPLETE MODEL PIPELINE
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

pipeline = Pipeline(
    steps=[
        (
            "preprocessing",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# ============================================================
# 9. FIT PIPELINE
# ============================================================

print("\n" + "=" * 60)
print("FITTING COMPLETE PIPELINE")
print("=" * 60)

pipeline.fit(X_train, y_train)

print("Pipeline fitted successfully.")


# ============================================================
# 10. EVALUATE
# ============================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC:  {roc_auc:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["<=50K", ">50K"]
    )
)


# ============================================================
# 11. CHECK PIPELINE STRUCTURE
# ============================================================

print("\n" + "=" * 60)
print("PIPELINE STRUCTURE")
print("=" * 60)

print(pipeline)


# ============================================================
# 12. SAVE PIPELINE
# ============================================================

import joblib

joblib.dump(
    pipeline,
    "adult_income_pipeline.pkl"
)

print("\nSaved complete pipeline as:")
print("adult_income_pipeline.pkl")