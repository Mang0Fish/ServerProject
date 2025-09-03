import dal
from models import *


def insert_user(user: User):
    dal.insert_user(user)