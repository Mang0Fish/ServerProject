from pydantic import BaseModel
import psycopg2
from models import *


def get_conn():
    return psycopg2.connect(
        dbname="ServerUsers",
        user="postgres",
        password="1234",
        host="localhost",
        port="5432"
    )





def create_table():
    with get_conn() as conn, conn.cursor() as cursor:
        cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        username TEXT PRIMARY KEY,
                        password TEXT NOT NULL,
                        tokens INTEGER NOT NULL
                    );
                """)
        conn.commit()


create_table()


def insert_user(user: User):
    with get_conn() as conn, conn.cursor() as cursor:
        cursor.execute(
            "INSERT INTO users (username, password, tokens) VALUES (%s, %s, %s)",
            (user.username, user.password, user.tokens)
        )
        conn.commit()
    return user


def get_users():
    with get_conn() as conn, conn.cursor() as cursor:
        cursor.execute("SELECT * FROM users")
        rows = cursor.fetchall()
        return [User(username=row[0], password=row[1], tokens=row[2]) for row in rows]


def get_user_by_username(username: str):
    with get_conn() as conn, conn.cursor() as cursor:
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        row = cursor.fetchone()
        if row:
            return User(username=row[0], password=row[1], tokens=row[2])
        else:
            return None
