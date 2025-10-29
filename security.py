import os
from datetime import datetime, timedelta, timezone
import jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
from dotenv import load_dotenv

load_dotenv()

SECRET = os.getenv("JWT_SECRET", "to_be_long_string")
ALG = os.getenv("ALG", "HS256")
ACCESS_MIN = int(os.getenv("ACCESS_MIN", "30"))
REFRESH_DAYS = int(os.getenv("REFRESH_DAYS", '1'))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def _now():
    return datetime.now(timezone.utc)


def create_access_token(username: str, role: str):
    now = _now()
    payload = {
        "sub": username,
        "iat": int(_now().timestamp()),
        "nbf": int((now - timedelta(seconds=5)).timestamp()),
        "exp": int((_now() + timedelta(minutes=ACCESS_MIN)).timestamp()),
        "role": role,
        "typ": "access"
    }
    return jwt.encode(payload, SECRET, algorithm=ALG)


def create_refresh_token(username: str):
    now = _now()
    payload = {
        "sub": username,
        "iat": int(now.timestamp()),
        "nbf": int((now - timedelta(seconds=5)).timestamp()),
        "exp": int((now + timedelta(days=REFRESH_DAYS)).timestamp()),
        "typ": "refresh"
    }
    return jwt.encode(payload, SECRET, algorithm=ALG)


def verify_read(token: str):
    try:
        return jwt.decode(token, SECRET, algorithms=[ALG], options={"require": ["exp", "iat", "sub", "typ", "role"]}, leeway=10)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


def get_current_user(token: str = Depends(oauth2_scheme)):
    data = verify_read(token)
    if data.get("typ") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Access token required")
    sub = data.get("sub")
    if not sub:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
    return {"username": data.get("sub"), "role": data.get("role", "user")}


def is_admin(current=Depends(get_current_user)):
    if current.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    return current







