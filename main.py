from fastapi import FastAPI, HTTPException, status, Depends
import psycopg2
from pydantic import BaseModel
from models import *
import bl
import os
from dotenv import load_dotenv
from security import get_current_user
from routers.token import router as token_router

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

# SECRET_KEY = os.getenv("SECRET_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

app = FastAPI(title="EZPredict", description="Learn and predict using various models", version='1.0')
app.include_router(token_router, prefix="/auth", tags=["auth"])


@app.get("/funcCheckingg")
def root(username: str, password: str):
    user = bl.verify_user(username, password)
    if not user:
        raise HTTPException(status_code=401, detail=f"Wrong Password or Username")
    return user


@app.post("/users/")
def create_item(user: UserCreate):
    try:
        bl.insert_user(user)
    except psycopg2.errors.UniqueViolation:  # username already exists
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User '{user.username}' already exists")
    return {"message": "User successfully added"}


@app.get("/users/")
def get_all():
    return bl.get_users()


@app.get("/users/{username}")
def read_user(username: str):
    user = bl.get_user_by_username(username)
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{username}' not found")
    return user


@app.put("/users/{username}")
def update_item(username: str, user: User):
    if username != user.username:
        raise HTTPException(status_code=400, detail="Username in path and body must match")
    found = bl.update_user(username, user)
    if not found:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User updated"}


@app.delete("/user/{username}")
def read_root(username: str):
    user = bl.delete_user(username)
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{username}' not found")
    return {"message": "User successfully deleted"}


@app.delete("/user_password/{username}")
def read_item(username: str, password: str):
    user = bl.verify_user(username, password)
    if not user:
        raise HTTPException(status_code=401, detail=f"Wrong Password or Username")
    bl.delete_user(username)
    return {"message": f"User {username} successfully deleted"}


@app.get("/tokens/{username}")
def read_item(username: str):
    tokens = bl.get_tokens(username)
    if not tokens:
        raise HTTPException(status_code=404, detail=f"User '{username}' not found")
    return tokens


@app.post("/add_tokens/{username}")
def create_item(username: str, payment: Payment):
    balance = bl.add_tokens(username, payment.amount)
    if not balance:
        raise HTTPException(status_code=404, detail=f"User '{username}' not found")
    return {f"User": username, "New balance": balance}


@app.get("/protected")
def protected_example(current=Depends(get_current_user)):
    return {"hello": current["username"]}


