import joblib
from datetime import datetime
from pathlib import Path
import os


def save_model(model):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"model_{timestamp}.pkl"

    path = "ml_models/" + name

    joblib.dump(model, path)

    return path


def load_model(path):
    return joblib.load(path)