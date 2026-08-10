# data_loader.py
"""
WHAT THIS DOES:
Loads the Adult Census Income dataset and does the FIRST pass of
real-data sanity checking — the checks you'd run on any new
dataset before trusting anything else about it.
"""
import pandas as pd
from sklearn.datasets import fetch_openml


def load_adult_income() -> pd.DataFrame:
    """
    Pulls the Adult Census Income dataset from OpenML.
    Returns features + target combined into one DataFrame for
    easier exploration (we'll split X/y later, once we understand
    what we're working with).
    """
    data = fetch_openml("adult", version=2, as_frame=True)
    df = data.frame
    # OpenML's target column is a categorical named "class" —
    # rename for clarity going forward
    df = df.rename(columns={"class": "income"})
    return df


def real_data_sanity_check(df: pd.DataFrame):
    """
    The checks that matter MORE on real data than on sklearn
    toy datasets, because real data actively hides problems
    that clean datasets never present.
    """
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print("=" * 60)

    print("\nDtypes:")
    print(df.dtypes)

    print("\n--- Missing values (pandas' own detection) ---")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "None detected by pandas")

    print("\n--- Hidden missing values (disguised as strings) ---")
    # This is the check that clean sklearn datasets never require —
    # real data often encodes "missing" as a string, not NaN, so
    # pandas' isnull() silently misses it entirely
    for col in df.select_dtypes(include=["object", "category"]).columns:
        suspicious = df[col].astype(str).isin(["?", "unknown", "Unknown", "", "N/A", "NA"])
        if suspicious.sum() > 0:
            print(f"  {col}: {suspicious.sum()} rows ({suspicious.mean()*100:.1f}%) "
                  f"contain a disguised missing value")

    print("\n--- Target class balance ---")
    print(df["income"].value_counts())
    print(df["income"].value_counts(normalize=True))

    print("\n--- Duplicate rows ---")
    print(f"{df.duplicated().sum()} exact duplicate rows")

    print("\n--- Cardinality of categorical columns ---")
    # High-cardinality categoricals need different encoding
    # treatment than low-cardinality ones — worth knowing counts
    # now, before Wednesday's encoding decisions
    for col in df.select_dtypes(include=["object", "category"]).columns:
        print(f"  {col}: {df[col].nunique()} unique values")