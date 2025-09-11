import dal
from models import *


def insert_user(user: User):
    dal.insert_user(user)


def get_users():
    return dal.get_users()


def get_user_by_username(username: str):
    return dal.get_user_by_username(username)


def update_user(username: str, user: User):
    return dal.update_user(username, user)


def delete_student(username: str):
    return dal.delete_user(username)


def get_tokens(username):
    return dal.get_tokens(username)


def add_tokens(username, amount):
   return dal.add_tokens(username, amount)