from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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

    # training
    model.fit(x_train, y_train)

    # evaluation
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    # return the model and accuracy
    return model, acc