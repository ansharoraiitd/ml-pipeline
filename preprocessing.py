# preprocessing.py
"""
WHAT THIS DOES:
Cleans the Adult Census Income dataset: drops redundant/non-
informative columns, handles missing values by TYPE (not
blanket imputation), and encodes categoricals by cardinality —
all with train-only fitting to avoid leakage.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are redundant or non-informative, based
    directly on what Monday's exploration found.
    """
    df = df.copy()

    # education-num already encodes education as an ordered integer —
    # keeping both lets a model implicitly double-count the same signal
    df = df.drop(columns=["education"])

    # fnlwgt is a census sampling weight, not a personal attribute —
    # it describes population representation, not the individual,
    # and including it risks the model learning a spurious pattern
    df = df.drop(columns=["fnlwgt"])

    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    "?" is MNAR for these columns (missingness itself is
    informative — see explanation above) — relabel rather than
    impute, so the model can learn from the missingness pattern
    directly instead of having it erased.
    """
    df = df.copy()
    cols_with_disguised_missing = ["workclass", "occupation", "native-country"]
    for col in cols_with_disguised_missing:
        df[col] = df[col].replace("?", "Unknown")
    return df


def encode_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Encode categoricals by cardinality, fitting ONLY on training
    data and applying that same fit to test data — the leakage-safe
    order of operations explained above.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Ordinal: genuine order exists
    ordinal_col = "education-num"  # already numeric/ordinal, no encoding needed
    # (kept here as a named exception, not silently skipped, so the
    # reasoning is visible rather than implicit)

    # Low cardinality (< 10 uniques): one-hot
    # High cardinality: frequency encoding (simple, leakage-safe version)
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns
    low_card = [c for c in cat_cols if X_train[c].nunique() < 10]
    high_card = [c for c in cat_cols if X_train[c].nunique() >= 10]

    print(f"Low-cardinality (one-hot): {low_card}")
    print(f"High-cardinality (frequency encoding): {high_card}")

    # One-hot: fit on train, apply same encoder to test
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(X_train[low_card])
    train_ohe = pd.DataFrame(
        ohe.transform(X_train[low_card]),
        columns=ohe.get_feature_names_out(low_card),
        index=X_train.index,
    )
    test_ohe = pd.DataFrame(
        ohe.transform(X_test[low_card]),
        columns=ohe.get_feature_names_out(low_card),
        index=X_test.index,
    )

    # Frequency encoding: compute frequencies from TRAIN ONLY,
    # then map those same frequencies onto test — this is the
    # leakage-safe version; a category that appears in test but
    # never in train gets mapped to 0 (handled explicitly, not
    # silently dropped)
    train_freq = X_train.copy()
    test_freq = X_test.copy()
    for col in high_card:
        freq_map = X_train[col].value_counts(normalize=True)
        train_freq[col] = X_train[col].astype(str).map(freq_map)
        test_freq[col] = X_test[col].astype(str).map(freq_map).fillna(0)

    X_train_final = pd.concat(
        [X_train.drop(columns=cat_cols), train_ohe, train_freq[high_card]], axis=1
    )
    X_test_final = pd.concat(
        [X_test.drop(columns=cat_cols), test_ohe, test_freq[high_card]], axis=1
    )

    return X_train_final, X_test_final