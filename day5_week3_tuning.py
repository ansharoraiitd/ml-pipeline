# day5_week3_tuning.py
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
from data_loader import load_adult_income
from preprocessing import clean_columns, handle_missing
from feature_pipeline import FeatureEngineer
from hyperparameter_tuning import run_grid_search, run_random_search, run_optuna_search

df = load_adult_income()
df = clean_columns(df)
df = handle_missing(df)

X = df.drop(columns=["income"])
y = (df["income"] == ">50K").astype(int)

# Same split as every prior day this week — X_test has NEVER been
# touched by anything, and won't be touched now either, until the
# very last step below
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


def build_full_pipeline():
    """Rebuild Wednesday's preprocessing, parameterized for reuse across searches."""
    numeric_cols = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    cat_cols = X_train.select_dtypes(include=["object"]).columns
    low_card = [c for c in cat_cols if X_train[c].nunique() < 10]
    high_card = [c for c in cat_cols if X_train[c].nunique() >= 10]

    preprocessor = ColumnTransformer(transformers=[
        ("numeric", StandardScaler(), numeric_cols + ["hours_x_education"]),
        ("low_card_cat", OneHotEncoder(handle_unknown="ignore"), low_card),
        ("high_card_cat", OneHotEncoder(handle_unknown="ignore"), high_card + ["age_bracket"]),
    ], remainder="drop")

    return Pipeline(steps=[
        ("feature_engineering", FeatureEngineer()),
        ("preprocessing", preprocessor),
        ("model", RandomForestClassifier(random_state=42)),
    ])


def build_optuna_pipeline(n_estimators, max_depth, min_samples_split):
    pipeline = build_full_pipeline()
    pipeline.set_params(
        model__n_estimators=n_estimators,
        model__max_depth=max_depth,
        model__min_samples_split=min_samples_split,
    )
    return pipeline


print("=" * 60)
print("1. GRID SEARCH")
print("=" * 60)
grid_search, grid_time = run_grid_search(build_full_pipeline(), X_train, y_train)

print("\n" + "=" * 60)
print("2. RANDOM SEARCH (same budget)")
print("=" * 60)
random_search, random_time = run_random_search(build_full_pipeline(), X_train, y_train)

print("\n" + "=" * 60)
print("3. OPTUNA (same budget)")
print("=" * 60)
optuna_study, optuna_time = run_optuna_search(build_optuna_pipeline, X_train, y_train)

print("\n" + "=" * 60)
print("SUMMARY: SAME BUDGET, THREE STRATEGIES")
print("=" * 60)
print(f"{'Method':<20}{'Best CV Score':<18}{'Time':<10}")
print(f"{'Grid Search':<20}{grid_search.best_score_:<18.4f}{grid_time:<10.1f}")
print(f"{'Random Search':<20}{random_search.best_score_:<18.4f}{random_time:<10.1f}")
print(f"{'Optuna':<20}{optuna_study.best_value:<18.4f}{optuna_time:<10.1f}")

# ------------------------------------------------------------------
# THE HONEST FINAL NUMBER — X_test touched here, for the first
# and only time, using whichever method won above
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("FINAL, HONEST TEST-SET EVALUATION (X_test touched for the first time)")
print("=" * 60)
best_search = max(
    [("Grid", grid_search.best_score_, grid_search.best_estimator_),
     ("Random", random_search.best_score_, random_search.best_estimator_)],
    key=lambda t: t[1],
)
method_name, cv_score, best_model = best_search

test_proba = best_model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, test_proba)

print(f"Winning method: {method_name} Search")
print(f"CV score (on X_train, used for selection): {cv_score:.4f}")
print(f"Test score (on X_test, never touched before now): {test_auc:.4f}")
print(f"Gap: {abs(cv_score - test_auc):.4f}  "
      f"(small gap = the CV estimate was honest; large gap = the "
      f"search may have overfit to X_train's cross-validation folds)")