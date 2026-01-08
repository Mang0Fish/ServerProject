from sklearn.model_selection import StratifiedKFold, cross_validate
import numpy as np

def cross_validate_classifier(pipeline, x, y, n_splits=5):
    """
        Runs Stratified K-Fold CV on an SKlearn Pipeline (preprocessor + classifier).
        Returns mean/std/per-fold for standard classification metrics.
        """

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scoring = {'accuracy': 'accuracy', 'f1': 'f1', 'precision': 'precision', 'recall': 'recall'}

    results = cross_validate(estimator=pipeline,
                             X=x,
                             y=y,
                             cv=cv,
                             scoring=scoring,
                             n_jobs=-1,
                             return_train_score=False)

    scores = {}
    for metric in scoring.keys():
        fold_scores = results[f'test_{metric}'].tolist()
        mean = np.mean(fold_scores)
        std = np.std(fold_scores)
        scores[metric] = {"mean":mean, "std":std, "per-fold":fold_scores}

    return {
        "method": "StratifiedKFold",
        "n_splits": n_splits,
        "shuffle": True,
        "random_state": 42,
        "scoring": scores,
    }