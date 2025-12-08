from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import SVC, SVR
from catboost import CatBoostClassifier, CatBoostRegressor

"""
accuracy score, F1, precision, recall = classification
r2, adjusted r2, MAE, MSE = regression

add to classification train_test_split: stratify = y 

...

"""


def train_linear_reg(x, y):
    # train test
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )

    # model type
    model = LinearRegression()

    # training
    model.fit(x_train, y_train)

    # evaluation
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)

    # return the model and accuracy
    return model, r2


def train_logistic_reg(x, y):
    # train test
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # model type
    model = LogisticRegression(max_iter=1000)
    """The default solver is 'lbfgs', if you want to change it to saga you can change 
    the penalty to l1 (only with saga), the default penalty is l2"""

    # training
    model.fit(x_train, y_train)

    # evaluation
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    # return the model and accuracy
    return model, acc


def train_random_forest_classifier(x, y):
    # train test
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # model type
    model = RandomForestClassifier(
        n_estimators=350,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    )

    # training
    model.fit(x_train, y_train)

    #
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    # return model and accuracy
    return model, acc


def train_random_forest_regressor(x, y):
    # train test
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )

    # model type
    model = RandomForestRegressor(
        n_estimators=350,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    )

    # training
    model.fit(x_train, y_train)

    #
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)

    # return model and accuracy
    return model, r2


def train_svm_classifier(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True # To predict probability with predict_proba
    )

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    return model, acc


def train_svm_regressor(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = SVR(
        kernel="rbf",
        C=1.0,
        gamma="scale",
    )

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)

    return model, r2


def train_catboost_classifier(x, y, categorical_cols=None):
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="Accuracy",
        verbose=False,
        early_stopping_rounds=50
    )

    model.fit(
        x_train, y_train,
        eval_set=(x_test, y_test),
        cat_features=categorical_cols
    )

    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    return model, acc


def train_catboost_regressor(x, y, categorical_cols=None):
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="RMSE",
        verbose=False,
        early_stopping_rounds=50
    )

    model.fit(
        x_train, y_train,
        eval_set=(x_test, y_test),
        cat_features=categorical_cols
    )

    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)

    return model, r2
