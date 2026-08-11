# day_sunday_week4_shap_preview.py
"""
WHAT THIS DOES:
Bare-minimum smoke test — confirm SHAP installs and runs against
Week 3's actual pipeline, before Monday's real explanation dive.
"""
import shap
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from data_loader import load_adult_income
from preprocessing import clean_columns, handle_missing
from feature_pipeline import FeatureEngineer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

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
    ("model", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
])
pipeline.fit(X_train, y_train)

print("Model trained. Running the smallest possible SHAP smoke test...")
X_test_transformed = pipeline[:-1].transform(X_test.iloc[:20])  # tiny sample, speed only
explainer = shap.TreeExplainer(pipeline.named_steps["model"])
shap_values = explainer.shap_values(X_test_transformed)

print(f"SHAP ran successfully. Output shape: {shap_values[1].shape if isinstance(shap_values, list) else shap_values.shape}")
print("(Full explanation of what this output means, and how to visualize it, is Monday's actual content.)")