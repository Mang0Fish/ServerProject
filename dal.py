import os

from pydantic import BaseModel
import psycopg2
from models import *

from dotenv import load_dotenv

"""
dbname="ServerUsers",
        user="postgres",
        password="1234",
        host="localhost",
        port="5432"
        
DELETE LATER
"""
load_dotenv()
def get_conn():
    dataSrc = os.getenv("DATABASE_URL")
    return psycopg2.connect(
        dataSrc
    )


def create_table():
    with get_conn() as conn, conn.cursor() as cursor:
        cursor.execute("""
                    CREATE TABLE IF NOT EXISTS public.users (
                        username TEXT PRIMARY KEY,
                        password TEXT NOT NULL,
                        tokens   INTEGER NOT NULL DEFAULT 0,                        
                        salt     TEXT NOT NULL,                        
                        role     TEXT NOT NULL DEFAULT 'user',
                        CONSTRAINT role_check CHECK (role IN ('user','admin'))
                    );
                """)
        conn.commit()


create_table()


def insert_user(user: UserCreate):
    with get_conn() as conn, conn.cursor() as cursor:
        cursor.execute(
            "INSERT INTO users (username, password, tokens) VALUES (%s, %s, 0)",
            (user.username, user.password)
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


def update_user(username: str, user: User):
    with get_conn() as conn, conn.cursor() as cursor:
        cursor.execute("UPDATE users SET password = %s, tokens = %s WHERE username = %s",
                       (user.password, user.tokens, username))
        conn.commit()
        user.username = username
        return cursor.rowcount == 1


def delete_user(username):
    with get_conn() as conn, conn.cursor() as cursor:
        cursor.execute("DELETE FROM users WHERE username = %s", (username,))
        conn.commit()
        return cursor.rowcount == 1


def get_tokens(username):
    with get_conn() as conn, conn.cursor() as cursor:
        cursor.execute("SELECT tokens FROM users WHERE username = %s", (username,))
        row = cursor.fetchone()
    return row[0] if row else None


def add_tokens(username, amount):
    with get_conn() as conn, conn.cursor() as cursor:
        cursor.execute("UPDATE users SET tokens = tokens + %s WHERE username = %s RETURNING tokens", (amount, username))
        row = cursor.fetchone()
        return row[0] if row else None
