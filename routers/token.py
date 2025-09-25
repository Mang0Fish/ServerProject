from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
import bl
import dal
from security import create_access_token, create_refresh_token, verify_read
from models import RefreshIn

router = APIRouter()


@router.post("/login")
def login(form: OAuth2PasswordRequestForm = Depends()):
    user = bl.verify_user(form.username, form.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Wrong username or password")
    access_token = create_access_token(username=user['username'], role=user['role'])
    refresh_token = create_refresh_token(username=user['username'])
    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}


@router.post("/auth/refresh")
def refresh(payload: RefreshIn):
    token = payload.refresh_token
    if not token:
        raise HTTPException(400, "Refresh token is required")

    try:
        data = verify_read(token)
    except Exception:
        raise HTTPException(401, "Invalid or expired token")

    if data.get("typ") != "refresh":
        raise HTTPException(401, "Wrong token type")

    username = data.get("sub")
    user = dal.get_user_auth(username)
    if not user:
        raise HTTPException(401, "User not found")

    access_token = create_access_token(username=user.username, role=user.role)
    # refresh_token = create_refresh_token(username=user.username)

    return {"access_token": access_token, "refresh_token": token, "token_type": "bearer"}
