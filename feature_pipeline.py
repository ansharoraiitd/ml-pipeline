# feature_pipeline.py
"""
WHAT THIS DOES:
Wraps Tuesday's manual preprocessing into a proper sklearn
Pipeline + ColumnTransformer (leakage-safe automatically, works
correctly inside cross-validation), and adds engineered features
(interaction term, binned age) on top.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that adds engineered features. Inheriting
    from BaseEstimator + TransformerMixin lets this slot directly
    into a Pipeline like any built-in sklearn step — fit/transform
    get called automatically at the right time, in the right order,
    inside every cross-validation fold.
    """

    def fit(self, X: pd.DataFrame, y=None):
        # No parameters need to be learned from data for these
        # particular engineered features (they're deterministic
        # functions of existing columns) — fit is a no-op here,
        # but the method must exist for Pipeline compatibility
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Interaction term: hours worked combined with education
        # level — captures the "long hours + advanced degree"
        # signal a linear model can't otherwise represent
        X["hours_x_education"] = X["hours-per-week"] * X["education-num"]

        # Binned age: captures non-linear age/income relationship
        # explicitly, as its own categorical feature
        X["age_bracket"] = pd.cut(
            X["age"],
            bins=[0, 25, 40, 60, 100],
            labels=["18-25", "26-40", "41-60", "60+"],
        ).astype(str)

        return X


def build_pipeline(low_card_cols, high_card_cols, numeric_cols) -> Pipeline:
    """
    Full leakage-safe pipeline: engineer features first, then
    route each column group through the right preprocessing,
    all fit correctly inside every CV fold automatically.
    """
    # ColumnTransformer applies different steps to different
    # column groups, in parallel, then concatenates the results
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_cols),
            ("low_card_cat", OneHotEncoder(handle_unknown="ignore"), low_card_cols),
            # High-cardinality columns here get one-hot too, for
            # pipeline simplicity today — frequency encoding inside
            # a leakage-safe Pipeline needs a custom transformer,
            # which you'll build in Week 5 alongside more advanced
            # encoding strategies
            ("high_card_cat", OneHotEncoder(handle_unknown="ignore"), high_card_cols),
        ],
        remainder="drop",  # explicit: anything not listed above is dropped,
                           # not silently passed through unprocessed
    )

    full_pipeline = Pipeline(steps=[
        ("feature_engineering", FeatureEngineer()),
        ("preprocessing", preprocessor),
    ])

    return full_pipeline