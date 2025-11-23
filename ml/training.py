from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

"""
accuracy score, F1, precision, recall  = classification
r2, adjusted r2, MAE, MSE = regression
"""


"""
def train_example_model():
    #  dataset with almost 1800 rows
    digs = load_digits()
    x = digs.data
    y = digs.target

    # train test
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )

    # model type
    model = LogisticRegression(max_iter=1000)

    # training
    model.fit(x_train, y_train)

    # evaluation
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    # return the model and accuracy
    return model, acc
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
        random_state=42
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


def train_random_forrest_classifier(x, y):
    # train test
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
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


def train_random_forrest_regressor(x, y):
    # train test
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
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
    r2 = r2_score(y_test, y_pred)

    # return model and accuracy
    return model, r2
