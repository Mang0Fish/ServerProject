from sklearn.model_selection import StratifiedKFold, cross_validate
import numpy as np
from catboost import Pool, cv

def cross_validate_classifier(pipeline, x, y, n_splits=5):
    """
        Runs Stratified K-Fold CV on an SKlearn Pipeline (preprocessor + classifier).
        Returns mean/std/per-fold for standard classification metrics.
        """

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scoring = {'accuracy': 'accuracy', 'f1': 'f1', 'precision': 'precision', 'recall': 'recall'}

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
        mean = np.mean(fold_scores)
        std = np.std(fold_scores)
        scores[metric] = {"mean":mean, "std":std, "per_fold":fold_scores}

    return {
        "method": "StratifiedKFold",
        "n_splits": n_splits,
        "shuffle": True,
        "random_state": 42,
        "scoring": scores,
    }


def cross_validate_catboost_classifier(x, y, cat_cols, params, n_splits=5):
    """
    CatBoost built-in cross validation. Uses the hyperparams the user chose.
    Returns the final iteration's test metric mean/std (and best iteration if available).
    """

    pool = Pool(data=x, label=y, cat_features=cat_cols)

    cv_results = cv(
        pool=pool,
        params=params,
        fold_count=n_splits,
        shuffle=True,
        partition_random_seed=42,
        stratified=True,
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
