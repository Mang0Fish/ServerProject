from fastapi import HTTPException
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, r2_score, precision_score, recall_score, f1_score,
                             mean_absolute_error, mean_squared_error, root_mean_squared_error,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.svm import SVC, SVR, LinearSVC
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.pipeline import Pipeline
from ml.preprocessing import build_preprocessor
from ml.validation import cross_validate_classifier, cross_validate_catboost, cross_validate_regressor
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

"""
accuracy score, F1, precision, recall = classification
r2, adjusted r2, MAE, MSE = regression

add to classification train_test_split: stratify = y 

...

"""
import numpy as np  # add at top if not present

def tune_threshold_binary_f1(y_true, y_prob, positive_label=None, steps=201):
    """
    Binary-only threshold tuning on a validation split.
    Returns best threshold by F1 + metrics at best vs default(0.5).

    y_true can be strings or numbers.
    y_prob is probability of the positive class (shape: [n_samples]).
    """

    if y_prob is None:
        return None

    labels = list(np.unique(y_true))
    if len(labels) != 2:
        return None

    # Pick a stable positive label if user didn't specify
    if positive_label is None:
        labels_sorted = sorted(labels, key=lambda v: str(v))
        positive_label = labels_sorted[1]  # treat "second" as positive

    y_true_bin = (np.array(y_true) == positive_label).astype(int)

    # Guard against degenerate cases
    if len(np.unique(y_true_bin)) != 2:
        return None

    def eval_at(th):
        pred = (np.array(y_prob) >= th).astype(int)
        p = precision_score(y_true_bin, pred, zero_division=0)
        r = recall_score(y_true_bin, pred, zero_division=0)
        f = f1_score(y_true_bin, pred, zero_division=0)
        cm = confusion_matrix(y_true_bin, pred)
        cm = [[int(v) for v in row] for row in cm.tolist()]
        return float(p), float(r), float(f), cm

    # default 0.5
    p05, r05, f05, cm05 = eval_at(0.5)

    best_t = 0.5
    best_f = -1.0
    best_pack = (p05, r05, f05, cm05)

    for th in np.linspace(0.0, 1.0, steps):
        p, r, f, cm = eval_at(float(th))
        if f > best_f:
            best_f = f
            best_t = float(th)
            best_pack = (p, r, f, cm)

    bp, br, bf, bcm = best_pack
    positive_label = str(positive_label)
    return {
        "positive_label": positive_label,
        "best_threshold": best_t,
        "metrics_at_best_threshold": {
            "precision": bp,
            "recall": br,
            "f1": bf,
            "confusion_matrix": bcm,
        },
        "default_threshold": 0.5,
        "metrics_at_default_0_5": {
            "precision": p05,
            "recall": r05,
            "f1": f05,
            "confusion_matrix": cm05,
        },
        "scan": {
            "steps": steps
        }
    }


def classifier_evaluation(y_test, y_pred, y_prob=None):
    is_binary = len(set(y_test)) == 2
    avg = "binary" if is_binary else "weighted"

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=avg, zero_division=0)
    recall = recall_score(y_test, y_pred, average=avg,zero_division=0)
    f1 = f1_score(y_test, y_pred, average=avg,zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "confusion_matrix": cm}

    if y_prob is not None and len(set(y_test)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        max_points = 20  #

        if len(fpr) > max_points:
            indices = np.linspace(0, len(fpr) - 1, max_points).astype(int)
            fpr = fpr[indices]
            tpr = tpr[indices]

        metrics["roc_curve"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        }

    return metrics

def regressor_evaluation(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    return {"r2":r2, "mae":mae, "mse":mse, "rmse":rmse}


def train_linear_reg(x, y, hyperparams, cat_cols, num_cols):
    params = {
        "fit_intercept": hyperparams.get("fit_intercept", True),
        "positive" : hyperparams.get("positive", False),
    }
    used_hyperparams = params.copy()

    preprocessor = build_preprocessor(cat_cols, num_cols, True)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression(**params))
    ])

    # train test
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )

    # training
    pipeline.fit(x_train, y_train)

    # evaluation
    y_pred = pipeline.predict(x_test)

    metrics = regressor_evaluation(y_test, y_pred)

    # cross validation
    cv = cross_validate_regressor(pipeline, x, y)

    metrics["cv"] = cv

    return pipeline, metrics, used_hyperparams


def train_logistic_reg(x, y, hyperparams, cat_cols, num_cols):
    params = {
        "max_iter": hyperparams.get("max_iter", 1000),
        "C": hyperparams.get("C", 1.0),
        "solver": hyperparams.get("solver", "lbfgs"),
        "penalty": hyperparams.get("penalty", "l2"),
    }
    used_hyperparams = params.copy()

    preprocessor = build_preprocessor(cat_cols, num_cols, True)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(**params))
    ])

    # train test
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # training
    pipeline.fit(x_train, y_train)

    # evaluation
    y_pred = pipeline.predict(x_test)

    y_prob = None
    if len(set(y_test)) == 2:
        probs = pipeline.predict_proba(x_test)
        if probs is not None and probs.shape[1] == 2:
            y_prob = probs[:, 1]

    metrics = classifier_evaluation(y_test, y_pred, y_prob)

    # cross validation
    cv = cross_validate_classifier(pipeline, x, y)

    metrics["cv"] = cv

    if y_prob is not None and len(set(y_test)) == 2:
        pos_label = hyperparams.get("positive_label", None)
        steps = hyperparams.get("threshold_steps", 201)

        metrics["threshold_tuning"] = tune_threshold_binary_f1(
            y_true=y_test,
            y_prob=y_prob,
            positive_label=pos_label,
            steps=steps
        )

    return pipeline, metrics, used_hyperparams


def train_random_forest_classifier(x, y, hyperparams, cat_cols, num_cols):
    params = {
        "n_estimators": hyperparams.get("n_estimators", 30),
        "max_depth": hyperparams.get("max_depth", None),
        "n_jobs": hyperparams.get("n_jobs", -1),
        "random_state": hyperparams.get("random_state", 42),
        "min_samples_split": hyperparams.get("min_samples_split", 2),
        "min_samples_leaf": hyperparams.get("min_samples_leaf", 1),
    }
    used_hyperparams = params.copy()

    preprocessor = build_preprocessor(cat_cols, num_cols, False)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(**params))
    ])

    # train test
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # training
    pipeline.fit(x_train, y_train)

    # training
    y_pred = pipeline.predict(x_test)

    y_prob = None
    if len(set(y_test)) == 2:
        probs = pipeline.predict_proba(x_test)
        if probs is not None and probs.shape[1] == 2:
            y_prob = probs[:, 1]

    metrics = classifier_evaluation(y_test, y_pred, y_prob)

    # cross validation
    cv = cross_validate_classifier(pipeline, x, y)

    metrics["cv"] = cv

    if y_prob is not None and len(set(y_test)) == 2:
        pos_label = hyperparams.get("positive_label", None)
        steps = hyperparams.get("threshold_steps", 201)

        metrics["threshold_tuning"] = tune_threshold_binary_f1(
            y_true=y_test,
            y_prob=y_prob,
            positive_label=pos_label,
            steps=steps
        )

    return pipeline, metrics, used_hyperparams


def train_random_forest_regressor(x, y, hyperparams, cat_cols, num_cols):
    params = {
        "n_estimators": hyperparams.get("n_estimators", 30),
        "max_depth": hyperparams.get("max_depth", None),
        "n_jobs": hyperparams.get("n_jobs", -1),
        "random_state": hyperparams.get("random_state", 42),
        "min_samples_split": hyperparams.get("min_samples_split", 2),
        "min_samples_leaf": hyperparams.get("min_samples_leaf", 1),
    }
    used_hyperparams = params.copy()

    preprocessor = build_preprocessor(cat_cols, num_cols, False)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(**params))
    ])

    # train test
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )

    # training
    pipeline.fit(x_train, y_train)

    #
    y_pred = pipeline.predict(x_test)

    metrics = regressor_evaluation(y_test, y_pred)

    # cross validation
    cv = cross_validate_regressor(pipeline, x, y)

    metrics["cv"] = cv

    return pipeline, metrics, used_hyperparams


def train_svm_classifier(x, y, hyperparams, cat_cols, num_cols):
    # Large dataset uses linear SVM (faster)
    use_linear = len(x) >= hyperparams.get("linear_threshold", 10000)

    preprocessor = build_preprocessor(cat_cols, num_cols, True)

    if use_linear:
        params = {
            "C": hyperparams.get("C", 1.0),
            "max_iter": hyperparams.get("max_iter", 5000),
        }
        used_hyperparams = params.copy()
        used_hyperparams["svm_variant"] = "LinearSVC"

        base = LinearSVC(**params)
        #Ables the use of ROC/AUC, as LinearSVC has no predict_proba
        calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=3)

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", calibrated)
        ])

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42, stratify=y
        )

        pipeline.fit(x_train, y_train)

        y_pred = pipeline.predict(x_test)

        y_prob = None
        if len(set(y_test)) == 2:
            probs = pipeline.predict_proba(x_test)
            if probs is not None and probs.shape[1] == 2:
                y_prob = probs[:, 1]

        metrics = classifier_evaluation(y_test, y_pred, y_prob)

        if y_prob is not None and len(set(y_test)) == 2:
            pos_label = hyperparams.get("positive_label", None)
            steps = hyperparams.get("threshold_steps", 201)

            metrics["threshold_tuning"] = tune_threshold_binary_f1(
                y_true=y_test,
                y_prob=y_prob,
                positive_label=pos_label,
                steps=steps
            )


        return pipeline, metrics, used_hyperparams

    # Small dataset keeps the kernel=SVC
    params = {
        "C": hyperparams.get("C", 1.0),
        "kernel": hyperparams.get("kernel", "rbf"),
        "gamma": hyperparams.get("gamma", "scale"),
        "degree": hyperparams.get("degree", 3),
        "probability": hyperparams.get("probability", True),
    }
    used_hyperparams = params.copy()
    used_hyperparams["svm_variant"] = "SVC"

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", SVC(**params))
    ])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)

    y_prob = None
    if params.get("probability", False) and len(set(y_test)) == 2:
        probs = pipeline.predict_proba(x_test)
        if probs is not None and probs.shape[1] == 2:
            y_prob = probs[:, 1]

    metrics = classifier_evaluation(y_test, y_pred, y_prob)

    if y_prob is not None and len(set(y_test)) == 2:
        pos_label = hyperparams.get("positive_label", None)
        steps = hyperparams.get("threshold_steps", 201)

        metrics["threshold_tuning"] = tune_threshold_binary_f1(
            y_true=y_test,
            y_prob=y_prob,
            positive_label=pos_label,
            steps=steps
        )

    return pipeline, metrics, used_hyperparams



def train_svm_regressor(x, y, hyperparams, cat_cols, num_cols):
    params = {
        "C": hyperparams.get("C", 1.0),
        "kernel": hyperparams.get("kernel", "rbf"),
        "gamma": hyperparams.get("gamma", "scale"),
        "degree": hyperparams.get("degree", 3),
    }
    used_hyperparams = params.copy()

    preprocessor = build_preprocessor(cat_cols, num_cols, True)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", SVR(**params))
    ])

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)

    metrics = regressor_evaluation(y_test, y_pred)

    # cross validation
    cv = cross_validate_regressor(pipeline, x, y)

    metrics["cv"] = cv

    return pipeline, metrics, used_hyperparams


def train_catboost_classifier(x, y, cat_cols, hyperparams):
    params = {
        "iterations": hyperparams.get("iterations", 500),
        "learning_rate": hyperparams.get("learning_rate", 0.05),
        "depth": hyperparams.get("depth", 6),
        "loss_function": hyperparams.get("loss_function", "Logloss"),
        "eval_metric": hyperparams.get("eval_metric", "Accuracy"),
        "early_stopping_rounds": hyperparams.get("early_stopping_rounds", None),
        "logging_level": "Silent"
    }


    n_classes = int(y.nunique())
    if n_classes > 2:
        raise HTTPException(
            status_code=422,
            detail=(
                f"CatBoost binary loss_function='{params.get('loss_function')}' can't train multiclass targets "
                f"({n_classes} classes). Set hyperparams.loss_function='MultiClass' "
                f"(and choose an appropriate eval_metric, e.g. 'Accuracy')."
            )
        )

    used_hyperparams = params.copy()

    if cat_cols:
        x = x.copy()
        x[cat_cols] = x[cat_cols].fillna("__MISSING__").astype(str)

    model = CatBoostClassifier(**params)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model.fit(
        x_train, y_train,
        eval_set=(x_test, y_test),
        cat_features=cat_cols,
        use_best_model = True
    )

    y_pred = model.predict(x_test)

    y_prob = None
    if len(set(y_test)) == 2:
        probs = model.predict_proba(x_test)
        if probs is not None and probs.shape[1] == 2:
            y_prob = probs[:, 1]

    metrics = classifier_evaluation(y_test, y_pred, y_prob)

    # cross validation
    cv = cross_validate_catboost(x, y, cat_cols, used_hyperparams, True)

    metrics["cv"] = cv
    if y_prob is not None and len(set(y_test)) == 2:
        pos_label = hyperparams.get("positive_label", None)
        steps = hyperparams.get("threshold_steps", 201)

        metrics["threshold_tuning"] = tune_threshold_binary_f1(
            y_true=y_test,
            y_prob=y_prob,
            positive_label=pos_label,
            steps=steps
        )

    return model, metrics, used_hyperparams


def train_catboost_regressor(x, y, cat_cols, hyperparams):
    params = {
        "iterations": hyperparams.get("iterations", 500),
        "learning_rate": hyperparams.get("learning_rate", 0.05),
        "depth": hyperparams.get("depth", 6),
        "loss_function": hyperparams.get("loss_function", "RMSE"),
        "eval_metric": hyperparams.get("eval_metric", "RMSE"),
        "early_stopping_rounds": hyperparams.get("early_stopping_rounds", None),
        "logging_level": "Silent"
    }
    used_hyperparams = params.copy()

    if cat_cols:
        x = x.copy()
        x[cat_cols] = x[cat_cols].fillna("__MISSING__").astype(str)

    model = CatBoostRegressor(**params)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    model.fit(
        x_train, y_train,
        eval_set=(x_test, y_test),
        cat_features=cat_cols,
        use_best_model = True
    )

    y_pred = model.predict(x_test)

    metrics = regressor_evaluation(y_test, y_pred)

    # cross validation
    cv = cross_validate_catboost(x, y, cat_cols, used_hyperparams, False)

    metrics["cv"] = cv

    return model, metrics, used_hyperparams
