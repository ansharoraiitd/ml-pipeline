# shap_explainability.py
"""
WHAT THIS DOES:
Applies SHAP to the tuned Week 3 pipeline — verifies the additive
property directly, then produces both local (single prediction)
and global (whole-dataset) explanations.
"""
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


def get_transformed_feature_names(pipeline) -> list:
    """
    After ColumnTransformer + OneHotEncoder, feature names change
    (e.g. 'sex' becomes 'sex_Male', 'sex_Female'). SHAP plots need
    the REAL post-transform names to be readable, not generic
    'feature_0', 'feature_1' labels.
    """
    preprocessor = pipeline.named_steps["preprocessing"]
    return list(preprocessor.get_feature_names_out())


def explain_predictions(pipeline, X_sample: pd.DataFrame):
    """
    Runs the pipeline's preprocessing manually (SHAP needs to work
    on the MODEL directly, after preprocessing, not on the raw
    pipeline object), then computes SHAP values via TreeExplainer.
    """
    X_transformed = pipeline[:-1].transform(X_sample)
    feature_names = get_transformed_feature_names(pipeline)

    model = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    # For binary classification, newer SHAP versions return
    # (rows, features, classes). We care about class 1 (>50K).
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]    

    return shap_values, X_transformed, feature_names, explainer.expected_value


def verify_additive_property(shap_values, expected_value, model, X_transformed, row_idx=0):
    """
    Directly checks: baseline + sum(shap values for this row) ==
    actual model prediction for this row. Proves the guarantee
    rather than just stating it.
    """
    base = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value
    shap_sum = shap_values[row_idx].sum()
    reconstructed = base + shap_sum

    actual_proba = model.predict_proba(X_transformed[row_idx:row_idx+1])[0, 1]

    print(f"Baseline (average) prediction:        {base:.4f}")
    print(f"Sum of this row's SHAP values:         {shap_sum:.4f}")
    print(f"Baseline + SHAP sum (reconstructed):   {reconstructed:.4f}")
    print(f"Actual model prediction for this row:  {actual_proba:.4f}")
    print(f"Match (within floating point): {np.isclose(reconstructed, actual_proba, atol=1e-4)}")


def plot_global_summary(shap_values, X_transformed, feature_names, filename="shap_summary.png"):
    """Global importance — aggregated from the same per-row values used locally."""
    plt.figure()
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches="tight")
    print(f"Saved {filename}")


def plot_local_waterfall(shap_values, expected_value, X_transformed, feature_names,
                          row_idx=0, filename="shap_waterfall_example.png"):
    """One specific prediction, explained — the 'why THIS person' answer."""
    base = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value
    explanation = shap.Explanation(
        values=shap_values[row_idx],
        base_values=base,
        data=X_transformed[row_idx],
        feature_names=feature_names,
    )
    plt.figure()
    shap.plots.waterfall(explanation, show=False, max_display=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches="tight")
    print(f"Saved {filename}")