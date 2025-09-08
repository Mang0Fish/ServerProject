from fastapi import FastAPI, HTTPException, status
import psycopg2
from pydantic import BaseModel
from models import *
import bl

"""
uvicorn main:app --reload --port 8000 

http://127.0.0.1:8000/docs



git add .
git commit -m "Enter a note here"
git push

"""

app = FastAPI(title="EZPredict", description="Learn and predict using various models", version='1.0')


@app.get("/")
def root():
    return {"message": "Hello FastAPI!"}


@app.post("/users/")
def create_item(user: User):
    try:
        bl.insert_user(user)
    except psycopg2.errors.UniqueViolation:  # username already exists
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User '{user.username}' already exists")
    return {"message": "User successfully added"}


@app.get("/users/")
def get_all():
    result = bl.get_users()
    return {"result": result}


@app.get("/users/{username}")
def read_item(username: str):
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


