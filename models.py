from pydantic import BaseModel, Field, StrictInt


class User(BaseModel):
    username: str
    password: str
    tokens: int

class Payment(BaseModel):
    credit_card: str = Field(pattern=r"^\d{4}-\d{4}-\d{4}-\d{4}$", description="Credit card valid pattern check")
    amount: StrictInt = Field(gt=0, description="Tokens must be a whole number greater than 0")

