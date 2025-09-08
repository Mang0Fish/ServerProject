from fastapi import FastAPI
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
    bl.insert_user(user)
    return {"message": "User successfully added"}


@app.get("/users/")
def get_all():
    result = bl.get_users()
    return {"result": result}


@app.get("/users/{username}")
def read_item(username: str):
    return bl.get_user_by_username(username)