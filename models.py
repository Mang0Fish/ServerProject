from pydantic import BaseModel


class User(BaseModel):
    username: str
    password: str
    tokens: int

class Payment(BaseModel):
    credit_card: str
    amount: int

