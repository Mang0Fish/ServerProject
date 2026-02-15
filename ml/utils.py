from enum import Enum

import joblib
from datetime import datetime
from fastapi import HTTPException
import pandas as pd
from io import StringIO
from ml.training import train_catboost_classifier, train_catboost_regressor, train_random_forest_classifier, \
    train_random_forest_regressor, train_svm_classifier, train_svm_regressor, train_linear_reg, train_logistic_reg
import json
import os
from pandas.api.types import is_numeric_dtype, is_categorical_dtype, is_string_dtype


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


def save_metadata(model_path, metadata):
    meta_path = model_path.replace(".pkl", ".meta.json")

    try:
        with open(meta_path, "w") as f:
            json.dump(metadata, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save metadata: {str(e)}")


def load_metadata(model_path):
    meta_path = model_path.replace(".pkl", ".meta.json")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Model does not exist")

    try:
        with open(meta_path) as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model metadata: {str(e)}")


def load_model(path):
    return joblib.load(path)


def validate_features(input_data, expected_data):
    input_features = set(input_data.keys())
    expected_features = set(expected_data.keys())

    missing = expected_features - input_features
    extra = input_features - expected_features

    if missing:
        raise HTTPException(status_code=422, detail=f"Missing features: {missing}")

    if extra:
        raise HTTPException(status_code=422, detail=f"Unexpected features: {extra}")

    for feature, typ in expected_data.items():
        value = input_data[feature]

        if typ == "numeric":
            if not isinstance(value, (int, float)):
                raise HTTPException(status_code=422, detail=f"Feature '{feature}' must be numeric")

        elif typ == "categorical":
            if not isinstance(value, str):
                raise HTTPException(status_code=422, detail=f"Feature '{feature}' must be a string")


def verify_file(data, label_column, hyperparams=None):
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

    counts = df[label_column].value_counts(dropna=False)
    min_count = int(counts.min())

    if min_count < 2:
        raise HTTPException(
            status_code=422,
            detail=f"Not enough samples per class for stratified split/CV. "
                   f"Smallest class has {min_count} row(s). Need at least 2 per class."
        )

    dropped_rows = 0
    if df[label_column].isna().any():
        n_missing = int(df[label_column].isna().sum())

        drop = False
        if hyperparams:
            drop = hyperparams.get("drop_missing_labels", False)

        if drop:
            dropped_rows = n_missing
            df = df.dropna(subset=[label_column])
            if len(df) < 2:
                raise HTTPException(
                    status_code=422,
                    detail="Not enough rows left after dropping missing labels."
                )
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Label column '{label_column}' contains {n_missing} missing value(s). "
                       f"Set drop_missing_labels=true in the hyperparams textbox to automatically drop those rows."
            )


    return df, dropped_rows


def train_model(df, label_column, model_type, hyperparams, dropped_rows = 0):
    if hyperparams is None:
        hyperparams = {}
    y = df[label_column]
    x = df.drop(columns=label_column)

    # cat_cols = [col for col in x.columns if x[col].dtype == "object"]
    # num_cols = [col for col in x.columns if x[col].dtype != "object"]

    num_cols = [col for col in x.columns if is_numeric_dtype(x[col])]
    cat_cols = [col for col in x.columns if col not in set(num_cols)]


    feature_types = {
        col: "categorical" if col in cat_cols else "numeric"
        for col in x.columns
    }

    if y.dtype == "object":
        problem_type = "classification"
    elif y.nunique() <= 10:
        problem_type = "classification"
    else:
        problem_type = "regression"

    model = None
    metrics = None
    used_hyperparams = {}

    match model_type:
        case ModelEnum.catboost:
            if problem_type == "classification":
                model, metrics, used_hyperparams = train_catboost_classifier(x, y, cat_cols, hyperparams)
            else:
                model, metrics, used_hyperparams = train_catboost_regressor(x, y, cat_cols, hyperparams)
        case ModelEnum.randomforest:
            if problem_type == "classification":
                model, metrics, used_hyperparams = train_random_forest_classifier(x, y, hyperparams, cat_cols, num_cols)
            else:
                model, metrics, used_hyperparams = train_random_forest_regressor(x, y, hyperparams, cat_cols, num_cols)
        case ModelEnum.svm:
            if problem_type == "classification":
                model, metrics, used_hyperparams = train_svm_classifier(x, y, hyperparams, cat_cols, num_cols)
            else:
                model, metrics, used_hyperparams = train_svm_regressor(x, y, hyperparams, cat_cols, num_cols)
        case ModelEnum.linearregression:
            model, metrics, used_hyperparams = train_linear_reg(x, y, hyperparams, cat_cols, num_cols)
        case ModelEnum.logisticregression:
            model, metrics, used_hyperparams = train_logistic_reg(x, y, hyperparams, cat_cols, num_cols)
        case _:
            raise HTTPException(422, f"Unknown model type: {model_type}")

    if model is None or metrics is None:
        raise HTTPException(500, "Model training failed")

    metadata = {
        "model_type": model_type.value,
        "problem_type": problem_type,
        "label_column": label_column,
        "features": x.columns.tolist(),
        "feature_types": feature_types,
        "categorical_features": cat_cols,
        "numerical_features": num_cols,
        "metrics": metrics,
        "hyperparams": used_hyperparams,
        "dropped_rows": dropped_rows
    }

    saved_path = save_model(model)
    save_metadata(saved_path, metadata)

    return {
        "msg": "Training completed",
        "rows": len(df),
        "problem_type": problem_type,
        "categorical_columns": cat_cols,
        "features": x.columns.tolist(),
        "label": label_column,
        "score": metrics,
        "model": saved_path
    }

