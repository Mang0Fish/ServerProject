from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
import bl
from security import create_access_token

router = APIRouter()


@router.post("/login")
def login(form: OAuth2PasswordRequestForm = Depends()):
    user = bl.verify_user(form.username, form.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Wrong username or password")
    token = create_access_token(username=user['username'], role=user['role'])
    return {"access_token": token, "token_type": "bearer"}
