from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import SVC, SVR
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

"""
accuracy score, F1, precision, recall = classification
r2, adjusted r2, MAE, MSE = regression

add to classification train_test_split: stratify = y 

...

"""


def train_linear_reg(x, y, hyperparams):
    params = {
        "fit_intercept": hyperparams.get("fit_intercept", True),
        "positive" : hyperparams.get("positive", False),
    }

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
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
    r2 = r2_score(y_test, y_pred)

    # return the model and accuracy
    return pipeline, r2


def train_logistic_reg(x, y, hyperparams):
    params = {
        "max_iter": hyperparams.get("max_iter", 1000),
        "C": hyperparams.get("C", 1.0),
        "solver": hyperparams.get("solver", "lbfgs"),
        "penalty": hyperparams.get("penalty", "l2"),
    }

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
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
    acc = accuracy_score(y_test, y_pred)

    # returns the pipeline and accuracy
    return pipeline, acc


def train_random_forest_classifier(x, y, hyperparams):
    params = {
        "n_estimators": hyperparams.get("n_estimators", 30),
        "max_depth": hyperparams.get("max_depth", None),
        "n_jobs": hyperparams.get("n_jobs", -1),
        "random_state": hyperparams.get("random_state", 42),
        "min_samples_split": hyperparams.get("min_samples_split", 2),
        "min_samples_leaf": hyperparams.get("min_samples_leaf", 1),
    }
    # model type
    model = RandomForestClassifier(**params)

    # train test
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # training
    model.fit(x_train, y_train)

    #
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    # return model and accuracy
    return model, acc


def train_random_forest_regressor(x, y, hyperparams):
    params = {
        "n_estimators": hyperparams.get("n_estimators", 30),
        "max_depth": hyperparams.get("max_depth", None),
        "n_jobs": hyperparams.get("n_jobs", -1),
        "random_state": hyperparams.get("random_state", 42),
        "min_samples_split": hyperparams.get("min_samples_split", 2),
        "min_samples_leaf": hyperparams.get("min_samples_leaf", 1),
    }

    # model type
    model = RandomForestRegressor(**params)

    # train test
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )

    # training
    model.fit(x_train, y_train)

    #
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)

    # return model and accuracy
    return model, r2


def train_svm_classifier(x, y, hyperparams):
    params = {
        "C": hyperparams.get("C", 1.0),
        "kernel": hyperparams.get("kernel", "rbf"),
        "gamma": hyperparams.get("gamma", "scale"),
        "degree": hyperparams.get("degree", 3),
        "probability": True,
    }

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(**params))
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
    acc = accuracy_score(y_test, y_pred)

    return pipeline, acc


def train_svm_regressor(x, y, hyperparams):
    params = {
        "C": hyperparams.get("C", 1.0),
        "kernel": hyperparams.get("kernel", "rbf"),
        "gamma": hyperparams.get("gamma", "scale"),
        "degree": hyperparams.get("degree", 3),
    }

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(**params))
    ])

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    r2 = r2_score(y_test, y_pred)

    return pipeline, r2

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
    acc = accuracy_score(y_test, y_pred)

    return model, acc


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
    r2 = r2_score(y_test, y_pred)

    return model, r2
