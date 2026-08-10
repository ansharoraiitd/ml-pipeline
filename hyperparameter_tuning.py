# hyperparameter_tuning.py
"""
WHAT THIS DOES:
Compares GridSearchCV, RandomizedSearchCV, and Optuna on the SAME
full pipeline from Wednesday — tuning happens entirely within
X_train via cross-validation, X_test is held out and touched
exactly once, at the end, for an honest final number.
"""
import time
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import optuna


def run_grid_search(pipeline: Pipeline, X_train, y_train, cv=5):
    """
    Exhaustive search — every combination in the grid, cross-
    validated. Small grid deliberately, so it finishes in
    reasonable time and the exhaustive-cost point is felt directly.
    """
    param_grid = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [5, 10, 15],
        "model__min_samples_split": [5, 10, 20],
    }
    n_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"Grid search: {n_combinations} combinations × {cv} folds = "
          f"{n_combinations * cv} total model fits")

    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    search = GridSearchCV(
        pipeline, param_grid, cv=kfold, scoring="roc_auc", n_jobs=-1
    )

    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"Best params: {search.best_params_}")
    print(f"Best CV score: {search.best_score_:.4f}")
    print(f"Time: {elapsed:.1f}s")
    return search, elapsed


def run_random_search(pipeline: Pipeline, X_train, y_train, n_iter=27, cv=5):
    """
    Same total budget as the grid (27 combos), but sampling from
    WIDER distributions instead of a fixed small grid — testing
    the actual claim that random search covers more distinct
    values per parameter within the same compute budget.
    """
    from scipy.stats import randint

    param_distributions = {
        "model__n_estimators": randint(30, 300),
        "model__max_depth": randint(3, 25),
        "model__min_samples_split": randint(2, 30),
    }
    print(f"Random search: {n_iter} sampled combinations × {cv} folds = "
          f"{n_iter * cv} total model fits (same budget as grid search)")

    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        pipeline, param_distributions, n_iter=n_iter, cv=kfold,
        scoring="roc_auc", random_state=42, n_jobs=-1
    )

    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"Best params: {search.best_params_}")
    print(f"Best CV score: {search.best_score_:.4f}")
    print(f"Time: {elapsed:.1f}s")
    return search, elapsed


def run_optuna_search(pipeline_builder, X_train, y_train, n_trials=27, cv=5):
    """
    Bayesian optimization — each trial's result informs where the
    NEXT trial searches, rather than every trial being independent
    like grid/random search.
    """
    from sklearn.model_selection import cross_val_score

    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 30, 300)
        max_depth = trial.suggest_int("max_depth", 3, 25)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 30)

        model = pipeline_builder(n_estimators, max_depth, min_samples_split)
        scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring="roc_auc")
        return scores.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)  # keep output readable
    study = optuna.create_study(direction="maximize")

    start = time.time()
    study.optimize(objective, n_trials=n_trials)
    elapsed = time.time() - start

    print(f"Best params: {study.best_params}")
    print(f"Best CV score: {study.best_value:.4f}")
    print(f"Time: {elapsed:.1f}s")
    return study, elapsed