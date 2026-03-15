from fastapi import FastAPI, HTTPException, status, Depends, Query, File, UploadFile, Body, Form
import psycopg2
from pydantic import BaseModel
from models import *
import bl
import os
from dotenv import load_dotenv
from security import get_current_user, is_admin
from routers.token import router as token_router
from security import create_access_token, create_refresh_token
from routers.token import generate_tokens_for_user
import pandas as pd
from io import StringIO
from ml.utils import load_model, verify_file, train_model, ModelEnum, load_metadata, validate_features, list_models
from typing import Optional, Dict, Any
import json
import logging


"""
uvicorn main:app --reload --port 8000 

http://127.0.0.1:8000/docs



git add .
git commit -m "Enter a note here"
git push

or 

git add .; git commit -m "quick update"; git push


git checkout main
git checkout -b feat/auth-jwt

"""

load_dotenv()

logging.basicConfig(filename="server.log",
                    level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")

# SECRET_KEY = os.getenv("SECRET_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

app = FastAPI(title="EZPredict", description="Learn and predict using various models", version='1.0')
app.include_router(token_router, prefix="/auth", tags=["auth"])

"""
@app.get("/funcCheckingg")
def root(username: str, password: str):
    user = bl.verify_user(username, password)
    if not user:
        raise HTTPException(status_code=401, detail=f"Wrong Password or Username")
    return user
"""


@app.post("/train")
async def train_csv(file: UploadFile = File(...),
                    label_column: str = Form(...),
                    model_type: ModelEnum = Query(..., description="Select the model"),
                    hyperparams: Optional[str] = Form(None),
                    current=Depends(get_current_user)
                    ):

    # Step 1: read csv file into a DataFrame
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "File must be a CSV")

    if hyperparams:
        try:
            hyperparams = json.loads(hyperparams)
        except json.JSONDecodeError:
            raise HTTPException(400, "hyperparams must be a valid JSON")
    else:
        hyperparams = {}

    data = await file.read()
    df, dropped_rows, label_column = verify_file(data, label_column, hyperparams)

    balance = bl.spend_tokens(current["username"], 1)
    if balance is None:
        logging.warning(f"User {current['username']} tried training without enough tokens")
        raise HTTPException(status_code=402, detail="Not enough tokens for training")

    model_trained = train_model(df, label_column, model_type, hyperparams, dropped_rows)
    logging.info(f"User {current['username']} trained model {model_type} with label {label_column}")
    return model_trained


@app.post("/predict/{model_name}")
def predict(model_name: str, input_data: Dict[str, Any] = Body(...), current=Depends(get_current_user)):
    model_path = f"ml_models/{model_name}.pkl"

    balance = bl.spend_tokens(current["username"], 5)
    if balance is None:
        logging.warning(f"User {current['username']} tried predicting without enough tokens")
        raise HTTPException(status_code=402, detail="Not enough tokens for prediction")

    metadata = load_metadata(model_path)

    validate_features(
        input_data=input_data,
        expected_data=metadata["feature_types"]
    )


    # model/pipeline loading
    try:
        model = load_model(model_path)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not load model")

    # converting input to df
    try:
        df = pd.DataFrame([input_data])
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid input format")

    # predicting
    try:
        prediction = model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

    response = {"prediction": prediction[0].item()} # fixed the iterable error

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df)
        response["probabilities"] = probs[0].tolist()

    logging.info(f"User {current['username']} made prediction using model {model_name}")
    return response


@app.get("/models")
def get_models():
    return list_models()


@app.post("/signup/")
def create_item(user: UserCreate):
    try:
        bl.insert_user(user)
    except psycopg2.errors.UniqueViolation:  # username already exists
        logging.warning(f"Registration failed: user {user.username} already exists")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User '{user.username}' already exists")

    logging.info(f"User {user.username} registered")
    return "message: user added! "
    # return generate_tokens_for_user(user.username, user.password)


@app.get("/admin/users/", response_model=list[UserOut])
def get_all(current=Depends(get_current_user)):
    if current.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    return bl.get_users()


@app.get("/admin/users/{username}", response_model=UserOut)
def read_user(username: str, current=Depends(get_current_user)):
    if current.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    user = bl.get_user_by_username(username)
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{username}' not found")
    return user


@app.delete("/admin/remove_user/{username}")
def read_root(username: str, current=Depends(get_current_user)):
    if current.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    user = bl.delete_user(username)
    if not user:
        logging.warning(f"Admin {current['username']} tried deleting missing user {username}")
        raise HTTPException(status_code=404, detail=f"User '{username}' not found")
    logging.info(f"Admin {current['username']} deleted user {username}")
    return {"message": "User successfully deleted"}


@app.delete("/remove_user")
def remove_user(user: UserCreate):
    verified = bl.verify_user(user.username, user.password)
    if not verified:
        raise HTTPException(
            status_code=401,
            detail="Wrong username or password"
        )
    bl.delete_user(user.username)
    logging.info(f"User {user.username} deleted")
    return {"message": f"User {user.username} successfully deleted"}


@app.get("/tokens/")
def read_item(current=Depends(get_current_user)):
    tokens = bl.get_tokens(current["username"])
    if tokens is None:
        raise HTTPException(status_code=404, detail=f"User '{current["username"]}' not found")
    return tokens


@app.post("/add_tokens/")
def create_item(payment: Payment, current=Depends(get_current_user)):
    balance = bl.add_tokens(current["username"], payment.amount)
    if not balance:
        logging.warning(f"Token add failed: user {current['username']} not found")
        raise HTTPException(status_code=404, detail=f"User '{current["username"]}' not found")
    logging.info(f"User {current['username']} added {payment.amount} tokens")
    return {f"User": current["username"], "New balance": balance}

"""
@app.get("/protected/me")
def protected_example(current=Depends(get_current_user)):
    return {"hello": current["username"], "role": current["role"]}
"""

