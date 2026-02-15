from sklearn.model_selection import StratifiedKFold, cross_validate, KFold
import numpy as np
from catboost import Pool, cv
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score


def choose_n_splits(x, y, problem_type):
    """
    Picks CV Folds using conventions.
    Problem type is classification or regression, True for classification, False for regression.
    """
    rows = len(x) # using x in case of edge cases

    k = 10 if rows>=50 else 5
    k = min(k, rows)

    if problem_type: # Classification
        min_class = int(y.value_counts().min())
        k = min(k, min_class)

    while k >= 2 and (rows/k) < 10:
        k-= 1

    return k if k>= 2 else None


def cross_validate_classifier(pipeline, x, y):
    """
        Runs Stratified K-Fold CV on an SKlearn Pipeline (preprocessor + classifier).
        Returns mean/std/per-fold for standard classification metrics.
        """

    n_splits = choose_n_splits(x, y, True)
    if n_splits is None:
        return None

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # scoring = {'accuracy': 'accuracy', 'f1': 'f1', 'precision': 'precision', 'recall': 'recall'}
    n_classes = int(y.nunique())

    if n_classes == 2:
        scoring = {
            "accuracy": "accuracy",
            "precision": make_scorer(precision_score, average="binary", zero_division=0),
            "recall": make_scorer(recall_score, average="binary", zero_division=0),
            "f1": make_scorer(f1_score, average="binary", zero_division=0),
        }

    else:
        scoring = {
            "accuracy": "accuracy",
            "precision": make_scorer(precision_score, average="weighted", zero_division=0),
            "recall": make_scorer(recall_score, average="weighted", zero_division=0),
            "f1": make_scorer(f1_score, average="weighted", zero_division=0),
        }

    results = cross_validate(estimator=pipeline,
                             X=x,
                             y=y,
                             cv=splitter,
                             scoring=scoring,
                             n_jobs=-1,
                             return_train_score=False)

    scores = {}
    for metric in scoring.keys():
        fold_scores = results[f'test_{metric}'].tolist()

        scores[metric] = {"mean":float(np.mean(fold_scores)),
                          "std":float(np.std(fold_scores)),
                          "per_fold":fold_scores}

    return {
        "method": "StratifiedKFold",
        "n_splits": n_splits,
        "shuffle": True,
        "random_state": 42,
        "scoring": scores,
    }


def cross_validate_regressor(pipeline, x, y):
    """
    Runs K-Fold CV on an SKlearn Pipeline (preprocessor + regressor).
    Returns mean/std/per-fold for standard regression metrics.
    """
    n_splits = choose_n_splits(x, y, False)
    if n_splits is None:
        return None

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    scoring = {"r2":"r2",
               "mae":"neg_mean_absolute_error",
               "mse":"neg_mean_squared_error",
               }

    results = cross_validate(estimator=pipeline,
                             X=x,
                             y=y,
                             cv=splitter,
                             scoring=scoring,
                             n_jobs=-1,
                             return_train_score=False)
    scores = {}
    for metric in scoring.keys():
        fold_scores = results[f'test_{metric}'].tolist()
        if metric in {"mae","mse"}:
            fold_scores = [-float(x) for x in fold_scores]
        else:
            fold_scores = [float(x) for x in fold_scores]

        scores[metric] = {
            "mean": float(np.mean(fold_scores)),
            "std": float(np.std(fold_scores)),
            "per_fold": fold_scores
        }

    return {
        "method": "KFold",
        "n_splits": n_splits,
        "shuffle": True,
        "random_state": 42,
        "scoring": scores,
    }


def cross_validate_catboost(x, y, cat_cols, params, cv_type):
    """
    CatBoost built-in cross validation. Uses the hyperparams the user chose.
    Returns the final iteration's test metric mean/std (and best iteration if available).
    cv_type: If true is classification, else regression
    """
    n_splits = choose_n_splits(x, y, cv_type)
    if n_splits is None:
        return None

    pool = Pool(data=x, label=y, cat_features=cat_cols)

    cv_results = cv(
        pool=pool,
        params=params,
        fold_count=n_splits,
        shuffle=True,
        partition_random_seed=42,
        stratified=cv_type,
        verbose=False)

    metric = params["eval_metric"]
    mean_col = f"test-{metric}-mean"
    std_col = f"test-{metric}-std"

    last_row = cv_results.iloc[-1]
    score_mean = float(last_row[mean_col])
    score_std = float(last_row[std_col])

    return {
        "method": "CatBoost_CV",
        "n_splits": n_splits,
        "shuffle": True,
        "random_state": 42,
        "loss_function": params["loss_function"],
        "eval_metric": metric,
        "score": {"mean": score_mean, "std":score_std}
    }
