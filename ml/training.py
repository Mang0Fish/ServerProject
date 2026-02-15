from fastapi import HTTPException
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, r2_score, precision_score, recall_score, f1_score,
                             mean_absolute_error, mean_squared_error, root_mean_squared_error,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.svm import SVC, SVR
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.pipeline import Pipeline
from ml.preprocessing import build_preprocessor
from ml.validation import cross_validate_classifier, cross_validate_catboost, cross_validate_regressor

"""
accuracy score, F1, precision, recall = classification
r2, adjusted r2, MAE, MSE = regression

add to classification train_test_split: stratify = y 

...

"""

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
    params = {
        "C": hyperparams.get("C", 1.0),
        "kernel": hyperparams.get("kernel", "rbf"),
        "gamma": hyperparams.get("gamma", "scale"),
        "degree": hyperparams.get("degree", 3),
        "probability": True,
    }
    used_hyperparams = params.copy()

    preprocessor = build_preprocessor(cat_cols, num_cols, True)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", SVC(**params))
    ])

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    # In case somehow probability = False

    y_prob = None
    if len(set(y_test)) == 2:
        probs = pipeline.predict_proba(x_test)
        if probs is not None and probs.shape[1] == 2:
            y_prob = probs[:, 1]

    metrics = classifier_evaluation(y_test, y_pred, y_prob)

    # cross validation
    cv = cross_validate_classifier(pipeline, x, y)

    metrics["cv"] = cv

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
