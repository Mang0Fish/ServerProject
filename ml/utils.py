from enum import Enum

import joblib
from datetime import datetime
from fastapi import HTTPException
import pandas as pd
from io import StringIO
from ml.training import train_catboost_classifier, train_catboost_regressor, train_random_forest_classifier, \
    train_random_forest_regressor, train_svm_classifier, train_svm_regressor, train_linear_reg, train_logistic_reg



class ModelEnum(str, Enum):
    catboost = "catboost"
    randomforest = "randomforest"
    svm = "svm"
    linearregression = "linearregression"
    logisticregression = "logisticregression"


def save_model(model):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"model_{timestamp}.pkl"

    path = "ml_models/" + name

    joblib.dump(model, path)

    return path


def load_model(path):
    return joblib.load(path)


def verify_file(data, label_column):
    if not data:
        raise HTTPException(400, "The dataset is empty")

    try:
        csv_string = data.decode("utf-8")
    except:
        raise HTTPException(400, "Failed to decode the file a UTF-8 CSV")

    df = pd.read_csv(StringIO(csv_string))

    if df.empty:
        raise HTTPException(400, "The dataset is empty")

    if label_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Label column '{label_column}' does not exist"
        )

    if len(df.columns) < 2:
        raise HTTPException(400, "The dataset must contain at least 2 columns")

    if df[label_column].nunique() < 2:
        raise HTTPException(400, "The label column must contain at least 2 unique values")

    if len(df) < 10:
        raise HTTPException(400, "The dataset is too small to train the model")

    return df


def train_model(df, label_column, model_type, hyperparams):
    if hyperparams is None:
        hyperparams = {}
    y = df[label_column]
    x = df.drop(columns=label_column)

    cat_cols = [i for i, col in enumerate(x.columns) if x[col].dtype == "object"]
    num_cols = [i for i, col in enumerate(x.columns) if x[col].dtype != "object"]

    if y.dtype == "object":
        problem_type = "classification"
    elif y.nunique() <= 10:
        problem_type = "classification"
    else:
        problem_type = "regression"

    model = None
    score = None

    match model_type:
        case ModelEnum.catboost:
            if problem_type == "classification":
                model, score = train_catboost_classifier(x, y, cat_cols, hyperparams)
            else:
                model, score = train_catboost_regressor(x, y, cat_cols, hyperparams)
        case ModelEnum.randomforest:
            if problem_type == "classification":
                model, score = train_random_forest_classifier(x, y, hyperparams, cat_cols, num_cols)
            else:
                model, score = train_random_forest_regressor(x, y, hyperparams, cat_cols, num_cols)
        case ModelEnum.svm:
            if problem_type == "classification":
                model, score = train_svm_classifier(x, y, hyperparams, cat_cols, num_cols)
            else:
                model, score = train_svm_regressor(x, y, hyperparams, cat_cols, num_cols)
        case ModelEnum.linearregression:
            model, score = train_linear_reg(x, y, hyperparams, cat_cols, num_cols)
        case ModelEnum.logisticregression:
            model, score = train_logistic_reg(x, y, hyperparams, cat_cols, num_cols)
        case _:
            raise HTTPException(400, f"Unknown model type: {model_type}")

    if model is None or score is None:
        raise HTTPException(500, "Model training failed")

    saved_path = save_model(model)

    return {
        "msg": "Training completed",
        "rows": len(df),
        "problem_type": problem_type,
        "categorical_columns": cat_cols,
        "features": x.columns.tolist(),
        "label": label_column,
        "score": score,
        "model": saved_path
    }

