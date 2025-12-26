from pydantic import BaseModel, Field, StrictInt
from typing import Dict, Any

class User(BaseModel):
    username: str
    password: str
    tokens: int
    salt: str
    role: str


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=15)
    password: str = Field(min_length=6, max_length=300)


class UserOut(BaseModel):
    username: str
    tokens: StrictInt


class Payment(BaseModel):
    credit_card: str = Field(pattern=r"^\d{4}-\d{4}-\d{4}-\d{4}$", description="Credit card valid pattern check")
    amount: StrictInt = Field(gt=0, description="Tokens must be a whole number greater than 0")


class RefreshIn(BaseModel):
    refresh_token: str


class PredictRequest(BaseModel):
    model_path: str
    input: Dict[str, Any]