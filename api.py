# api.py
"""
WHAT THIS DOES:
FastAPI service wrapping the trained pipeline + SHAP explainer.
Trains ONCE at startup (not per-request), exposes prediction +
explanation via a single endpoint.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

import shap

from data_loader import load_adult_income
from preprocessing import clean_columns, handle_missing
from feature_pipeline import FeatureEngineer
from shap_explainability import get_transformed_feature_names


app = FastAPI(
    title="Income Prediction API",
    description=(
        "Predicts >$50K income likelihood from census-style attributes, "
        "with a per-prediction SHAP explanation."
    ),
    version="1.0.0",
)


# ------------------------------------------------------------------
# STARTUP: train once, hold in memory
# ------------------------------------------------------------------

print("Loading and training model (runs once, at startup)...")

df = load_adult_income()
df = clean_columns(df)
df = handle_missing(df)

X = df.drop(columns=["income"])
y = (df["income"] == ">50K").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


numeric_cols = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

cat_cols = X_train.select_dtypes(include=["object"]).columns

low_card = [
    c for c in cat_cols
    if X_train[c].nunique() < 10
]

high_card = [
    c for c in cat_cols
    if X_train[c].nunique() >= 10
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_cols + ["hours_x_education"],
        ),
        (
            "low_card_cat",
            OneHotEncoder(handle_unknown="ignore"),
            low_card,
        ),
        (
            "high_card_cat",
            OneHotEncoder(handle_unknown="ignore"),
            high_card + ["age_bracket"],
        ),
    ],
    remainder="drop",
)


pipeline = Pipeline(
    steps=[
        ("feature_engineering", FeatureEngineer()),
        ("preprocessing", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                random_state=42,
            ),
        ),
    ]
)


pipeline.fit(X_train, y_train)


# ------------------------------------------------------------------
# SHAP EXPLAINER
# ------------------------------------------------------------------

model = pipeline.named_steps["model"]

explainer = shap.TreeExplainer(model)

feature_names = get_transformed_feature_names(pipeline)

expected_value = explainer.expected_value

# Keep the class-1 expected value for binary classification
if isinstance(expected_value, (list, np.ndarray)):
    expected_value = np.asarray(expected_value).flatten()

    if len(expected_value) > 1:
        expected_value = expected_value[1]
    else:
        expected_value = expected_value[0]


print(
    f"Model ready. Test accuracy: "
    f"{pipeline.score(X_test, y_test):.4f}"
)


# ------------------------------------------------------------------
# REQUEST / RESPONSE SCHEMAS
# ------------------------------------------------------------------

class PersonInput(BaseModel):
    age: int
    workclass: str
    occupation: str
    education_num: int
    marital_status: str
    relationship: str
    race: str
    sex: str
    capital_gain: float = 0
    capital_loss: float = 0
    hours_per_week: int
    native_country: str = "United-States"


class Contribution(BaseModel):
    feature: str
    value: float


class PredictionResponse(BaseModel):
    prediction: str
    probability_above_50k: float
    top_contributing_factors: List[Contribution]


# ------------------------------------------------------------------
# ENDPOINTS
# ------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Income Prediction API",
        "docs": "/docs",
        "endpoints": ["/health", "/predict"],
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_test_accuracy": round(
            pipeline.score(X_test, y_test),
            4,
        ),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(person: PersonInput):
    """
    Predict income class AND explain the prediction via SHAP.
    """

    # --------------------------------------------------------------
    # Build a single-row DataFrame matching the training schema
    # --------------------------------------------------------------

    row = pd.DataFrame(
        [
            {
                "age": person.age,
                "workclass": person.workclass,
                "fnlwgt": 0,
                "education-num": person.education_num,
                "marital-status": person.marital_status,
                "occupation": person.occupation,
                "relationship": person.relationship,
                "race": person.race,
                "sex": person.sex,
                "capital-gain": person.capital_gain,
                "capital-loss": person.capital_loss,
                "hours-per-week": person.hours_per_week,
                "native-country": person.native_country,
            }
        ]
    )

    # API input is already in cleaned feature format.
    row = handle_missing(row)


    # --------------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------------

    proba = pipeline.predict_proba(row)[0, 1]

    prediction_label = (
        ">50K"
        if proba >= 0.5
        else "<=50K"
    )


    # --------------------------------------------------------------
    # SHAP EXPLANATION
    # --------------------------------------------------------------

    # Transform row using everything before the model
    row_transformed = pipeline[:-1].transform(row)

    # Calculate SHAP values
    raw_shap_vals = explainer.shap_values(row_transformed)


    # --------------------------------------------------------------
    # HANDLE DIFFERENT SHAP OUTPUT FORMATS
    #
    # Depending on the SHAP version, binary classification can give:
    #
    #   old SHAP:
    #       [class_0, class_1]
    #
    #   newer SHAP:
    #       (rows, features, classes)
    #
    #   or:
    #       (rows, features)
    #
    # We want:
    #
    #       (features,)
    # --------------------------------------------------------------

    if isinstance(raw_shap_vals, list):

        # Old SHAP format:
        # [class_0_values, class_1_values]

        shap_vals = raw_shap_vals[1]

    else:

        shap_vals = np.asarray(raw_shap_vals)

        if shap_vals.ndim == 3:
            # Shape:
            # (samples, features, classes)
            #
            # We want class 1.

            shap_vals = shap_vals[:, :, 1]

        elif shap_vals.ndim == 2:
            # Shape:
            # (samples, features)
            #
            # Already correct.

            pass

        elif shap_vals.ndim == 1:
            # Already:
            # (features,)

            pass

        else:
            raise ValueError(
                f"Unexpected SHAP output shape: {shap_vals.shape}"
            )


    # We are explaining exactly ONE row
    shap_vals = np.asarray(shap_vals)

    if shap_vals.ndim == 2:
        shap_vals = shap_vals[0]

    # Final sanity check
    if shap_vals.ndim != 1:
        raise ValueError(
            f"SHAP values should be 1D after processing, "
            f"but got shape {shap_vals.shape}"
        )


    # --------------------------------------------------------------
    # TOP 5 CONTRIBUTING FEATURES
    # --------------------------------------------------------------

    top_indices = np.argsort(
        np.abs(shap_vals)
    )[::-1][:5]


    top_factors = []

    for i in top_indices:

        i = int(i)

        top_factors.append(
            Contribution(
                feature=feature_names[i],
                value=round(
                    float(shap_vals[i]),
                    4,
                ),
            )
        )


    # --------------------------------------------------------------
    # RETURN RESPONSE
    # --------------------------------------------------------------

    return PredictionResponse(
        prediction=prediction_label,
        probability_above_50k=round(
            float(proba),
            4,
        ),
        top_contributing_factors=top_factors,
    )