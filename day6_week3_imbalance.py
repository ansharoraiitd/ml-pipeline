# day6_week3_imbalance.py
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from data_loader import load_adult_income
from preprocessing import clean_columns, handle_missing
from feature_pipeline import FeatureEngineer
from imbalance_handling import compare_imbalance_strategies, threshold_analysis

df = load_adult_income()
df = clean_columns(df)
df = handle_missing(df)

X = df.drop(columns=["income"])
y = (df["income"] == ">50K").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("=" * 60)
print(f"Class balance in training data: {y_train.value_counts(normalize=True).to_dict()}")
print("=" * 60)


def build_preprocessing():
    numeric_cols = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    cat_cols = X_train.select_dtypes(include=["object"]).columns
    low_card = [c for c in cat_cols if X_train[c].nunique() < 10]
    high_card = [c for c in cat_cols if X_train[c].nunique() >= 10]

    preprocessor = ColumnTransformer(transformers=[
        ("numeric", StandardScaler(), numeric_cols + ["hours_x_education"]),
        ("low_card_cat", OneHotEncoder(handle_unknown="ignore"), low_card),
        ("high_card_cat", OneHotEncoder(handle_unknown="ignore"), high_card + ["age_bracket"]),
    ], remainder="drop")

    from sklearn.pipeline import Pipeline as SkPipeline
    return SkPipeline(steps=[
        ("feature_engineering", FeatureEngineer()),
        ("preprocessing", preprocessor),
    ])


print("\n" + "=" * 60)
print("COMPARING IMBALANCE STRATEGIES (PR-AUC, cross-validated on X_train only)")
print("=" * 60)
results = compare_imbalance_strategies(build_preprocessing(), X_train, y_train)

# Pick the winning strategy for threshold analysis — using
# cross_val_predict so every prediction comes from a fold where
# that row was held out, still leakage-safe
print("\n" + "=" * 60)
print("THRESHOLD ANALYSIS on the best-performing strategy")
print("=" * 60)

# build the preprocessing components directly so that the
# imblearn Pipeline is flat and compatible with cross_val_predict

numeric_cols = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]

cat_cols = X_train.select_dtypes(include=["object"]).columns
low_card = [c for c in cat_cols if X_train[c].nunique() < 10]
high_card = [c for c in cat_cols if X_train[c].nunique() >= 10]

preprocessor = ColumnTransformer(transformers=[
    (
        "numeric",
        StandardScaler(),
        numeric_cols + ["hours_x_education"]
    ),
    (
        "low_card_cat",
        OneHotEncoder(handle_unknown="ignore"),
        low_card
    ),
    (
        "high_card_cat",
        OneHotEncoder(handle_unknown="ignore"),
        high_card + ["age_bracket"]
    ),
], remainder="drop")

# Baseline was the best strategy in the actual CV results,
# so threshold tuning should be performed on the baseline model.
best_pipeline = ImbPipeline(steps=[
    ("feature_engineering", FeatureEngineer()),
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )),
])

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

y_proba_cv = cross_val_predict(
    best_pipeline,
    X_train,
    y_train,
    cv=kfold,
    method="predict_proba"
)[:, 1]

threshold_analysis(y_train, y_proba_cv)