# imbalance_handling.py
"""
WHAT THIS DOES:
Compares class weighting, SMOTE (leakage-safe, inside a pipeline),
and threshold tuning on the Adult Income dataset — three genuinely
different fixes for the same underlying problem, evaluated
honestly with PR-AUC as the primary metric given the imbalance.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, precision_recall_curve,
    precision_score, recall_score, f1_score
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def compare_imbalance_strategies(preprocessing_pipeline, X_train, y_train, cv=5):
    """
    Baseline (no fix) vs class weighting vs SMOTE.

    Preprocessing and feature engineering happen inside the same
    imbalanced-learn pipeline so that every transformation is fitted
    separately inside each CV fold.
    """
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)

    feature_engineering = preprocessing_pipeline.named_steps["feature_engineering"]
    preprocessing = preprocessing_pipeline.named_steps["preprocessing"]

    strategies = {}

    # ------------------------------------------------------------
    # 1. BASELINE
    # ------------------------------------------------------------
    strategies["Baseline (no fix)"] = ImbPipeline(steps=[
        ("feature_engineering", feature_engineering),
        ("preprocessing", preprocessing),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )),
    ])

    # ------------------------------------------------------------
    # 2. CLASS WEIGHTING
    # ------------------------------------------------------------
    strategies["Class weighting"] = ImbPipeline(steps=[
        ("feature_engineering", feature_engineering),
        ("preprocessing", preprocessing),
        ("model", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42
        )),
    ])

    # ------------------------------------------------------------
    # 3. SMOTE
    # ------------------------------------------------------------
    strategies["SMOTE"] = ImbPipeline(steps=[
        ("feature_engineering", feature_engineering),
        ("preprocessing", preprocessing),
        ("smote", SMOTE(random_state=42)),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )),
    ])

    results = {}

    for name, pipeline in strategies.items():
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=kfold,
            scoring="average_precision"
        )

        results[name] = scores

        print(
            f"{name}: PR-AUC = "
            f"{scores.mean():.4f} ± {scores.std():.4f}"
        )

    return results

def threshold_analysis(y_true, y_proba, thresholds=None):
    """
    Sweep the classification threshold and show precision/recall/F1
    at each point — makes the "0.5 is arbitrary" point concrete,
    with actual numbers instead of just an assertion.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)

    results = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        results.append({
            "threshold": t,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        })

    thresholds_arr = [r["threshold"] for r in results]
    precisions = [r["precision"] for r in results]
    recalls = [r["recall"] for r in results]
    f1s = [r["f1"] for r in results]

    plt.figure(figsize=(9, 5))
    plt.plot(thresholds_arr, precisions, marker="o", label="Precision", markersize=4)
    plt.plot(thresholds_arr, recalls, marker="o", label="Recall", markersize=4)
    plt.plot(thresholds_arr, f1s, marker="o", label="F1", markersize=4)
    plt.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="default (0.5)")
    plt.xlabel("Classification threshold")
    plt.ylabel("Score")
    plt.title("Precision / Recall / F1 vs Classification Threshold")
    plt.legend()
    plt.savefig("threshold_analysis.png")
    print("Saved threshold_analysis.png")

    best_f1_idx = int(np.argmax(f1s))
    print(f"\nDefault threshold (0.5): "
          f"precision={results[10]['precision']:.3f}, recall={results[10]['recall']:.3f}")
    print(f"Best F1 threshold ({thresholds_arr[best_f1_idx]:.2f}): "
          f"precision={precisions[best_f1_idx]:.3f}, recall={recalls[best_f1_idx]:.3f}, "
          f"F1={f1s[best_f1_idx]:.3f}")

    return results