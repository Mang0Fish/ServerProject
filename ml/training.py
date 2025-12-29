from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, r2_score, precision_score, recall_score, f1_score,
                             mean_absolute_error, mean_squared_error, root_mean_squared_error,
                             confusion_matrix)
from sklearn.svm import SVC, SVR
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ml.preprocessing import build_preprocessor

"""
accuracy score, F1, precision, recall = classification
r2, adjusted r2, MAE, MSE = regression

add to classification train_test_split: stratify = y 

...

"""

def classifier_evaluation(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "confusion matrix": cm}


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

    metrics = classifier_evaluation(y_test, y_pred)

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

    metrics = classifier_evaluation(y_test, y_pred)

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

    metrics = classifier_evaluation(y_test, y_pred)

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

    return pipeline, metrics, used_hyperparams

# Cat cols is no longer = None, make sure there are no errors
def train_catboost_classifier(x, y, categorical_cols, hyperparams):
    params = {
        "iterations": hyperparams.get("iterations", 500),
        "learning_rate": hyperparams.get("learning_rate", 0.05),
        "depth": hyperparams.get("depth", 6),
        "loss_function": hyperparams.get("loss_function", "Logloss"),
        "eval_metric": hyperparams.get("eval_metric", "Accuracy"),
        "early_stopping_rounds": hyperparams.get("early_stopping_rounds", None),
        "verbose": False,
    }
    used_hyperparams = params.copy()

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
        cat_features=categorical_cols,
        use_best_model = True
    )

    y_pred = model.predict(x_test)

    metrics = classifier_evaluation(y_test, y_pred)

    return model, metrics, used_hyperparams


def train_catboost_regressor(x, y, categorical_cols, hyperparams):
    params = {
        "iterations": hyperparams.get("iterations", 500),
        "learning_rate": hyperparams.get("learning_rate", 0.05),
        "depth": hyperparams.get("depth", 6),
        "loss_function": hyperparams.get("loss_function", "RMSE"),
        "eval_metric": hyperparams.get("eval_metric", "RMSE"),
        "early_stopping_rounds": hyperparams.get("early_stopping_rounds", None),
        "verbose": False,
    }
    used_hyperparams = params.copy()

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
        cat_features=categorical_cols,
        use_best_model = True
    )

    y_pred = model.predict(x_test)

    metrics = regressor_evaluation(y_test, y_pred)

    return model, metrics, used_hyperparams
