import dal
from models import *


def insert_user(user: User):
    dal.insert_user(user)


def get_users():
    return dal.get_users()


def get_user_by_username(username: str):
    return dal.get_user_by_username(username)
