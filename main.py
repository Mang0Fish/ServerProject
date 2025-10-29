from fastapi import FastAPI, HTTPException, status, Depends
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
    return "message: user added! "
    # return generate_tokens_for_user(user.username, user.password)


@app.get("/admin/users/", response_model=list[UserOut])
def get_all(current=Depends(get_current_user)):
    return bl.get_users()


@app.get("/admin/users/{username}", response_model=UserOut)
def read_user(username: str, current=Depends(get_current_user)):
    user = bl.get_user_by_username(username)
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{username}' not found")
    return user


"""
@app.put("/users/{username}")  # updates password with no hashing (just delete this later)
def update_item(username: str, user: User):
    if username != user.username:
        raise HTTPException(status_code=400, detail="Username in path and body must match")
    found = bl.update_user(username, user)
    if not found:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User updated"}
"""
# possible password update


@app.delete("/admin/user/{username}")
def read_root(username: str, current=Depends(get_current_user)):
    if current.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    user = bl.delete_user(username)
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{username}' not found")
    return {"message": "User successfully deleted"}


@app.delete("/user_password/{username}")  # Not safe but a project requirement
def read_item(username: str, password: str):
    user = bl.verify_user(username, password)
    if not user:
        raise HTTPException(status_code=401, detail=f"Wrong Password or Username")
    bl.delete_user(username)
    return {"message": f"User {username} successfully deleted"}


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
        raise HTTPException(status_code=404, detail=f"User '{current["username"]}' not found")
    return {f"User": current["username"], "New balance": balance}


@app.get("/protected/me")
def protected_example(current=Depends(get_current_user)):
    return {"hello": current["username"], "role": current["role"]}


