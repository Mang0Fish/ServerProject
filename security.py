import os
from datetime import datetime, timedelta, timezone
import jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer

SECRET = os.getenv("JWT_SECRET", "to_be_long_string")
ALG = "HS256"
ACCESS_MIN = int(os.getenv("JWT_ACCESS_MIN", "15"))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")  # maybe false


def _now():
    return datetime.now(timezone.utc)


def create_access_token(username: str):
    payload = {
        "sub":username,
        "iat": int(_now().timestamp()),
        "exp": int((_now() + timedelta(minutes=ACCESS_MIN)).timestamp()),
    }
    return jwt.encode(payload, SECRET, algorithm=ALG)


def verify_read(token: str):
    try:
        return jwt.decode(token, SECRET, algorithms=[ALG])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


def get_current_user(token: str = Depends(oauth2_scheme)):
    data = verify_read(token)
    return {"username": data.get("sub")}




