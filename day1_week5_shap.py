# day1_week5_shap.py
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from data_loader import load_adult_income
from preprocessing import clean_columns, handle_missing
from feature_pipeline import FeatureEngineer
from shap_explainability import (
    explain_predictions, verify_additive_property,
    plot_global_summary, plot_local_waterfall
)

df = load_adult_income()
df = clean_columns(df)
df = handle_missing(df)
X = df.drop(columns=["income"])
y = (df["income"] == ">50K").astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_cols = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
cat_cols = X_train.select_dtypes(include=["object"]).columns
low_card = [c for c in cat_cols if X_train[c].nunique() < 10]
high_card = [c for c in cat_cols if X_train[c].nunique() >= 10]

preprocessor = ColumnTransformer(transformers=[
    ("numeric", StandardScaler(), numeric_cols + ["hours_x_education"]),
    ("low_card_cat", OneHotEncoder(handle_unknown="ignore"), low_card),
    ("high_card_cat", OneHotEncoder(handle_unknown="ignore"), high_card + ["age_bracket"]),
], remainder="drop")

pipeline = Pipeline(steps=[
    ("feature_engineering", FeatureEngineer()),
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)),
])
pipeline.fit(X_train, y_train)
print(f"Model trained. Test accuracy: {pipeline.score(X_test, y_test):.4f}")

# Use a manageable sample for SHAP — 200 rows is plenty to see
# real patterns without a long runtime
X_sample = X_test.iloc[:200].reset_index(drop=True)

print("\n" + "=" * 60)
print("COMPUTING SHAP VALUES")
print("=" * 60)
shap_values, X_transformed, feature_names, expected_value = explain_predictions(pipeline, X_sample)
print(f"SHAP values shape: {shap_values.shape}  (200 rows x {shap_values.shape[1]} features)")

print("\n" + "=" * 60)
print("VERIFYING THE ADDITIVE PROPERTY (row 0)")
print("=" * 60)
verify_additive_property(shap_values, expected_value, pipeline.named_steps["model"], X_transformed, row_idx=0)

print("\n" + "=" * 60)
print("GLOBAL EXPLANATION — feature importance from the same SHAP values")
print("=" * 60)
plot_global_summary(shap_values, X_transformed, feature_names)

print("\n" + "=" * 60)
print("LOCAL EXPLANATION — one specific person's prediction")
print("=" * 60)
row_to_explain = 0
print(f"Explaining row {row_to_explain}:")
print(X_sample.iloc[row_to_explain])
plot_local_waterfall(shap_values, expected_value, X_transformed, feature_names, row_idx=row_to_explain)