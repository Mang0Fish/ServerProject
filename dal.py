import os

from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from pydantic import BaseModel
import psycopg2
from models import *
from auth import hash_password, verify_password
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
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

def create_database():
    # IMPORTANT: connect to default postgres DB, not ServerUsers
    conn = psycopg2.connect(
        dbname="postgres",
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

    with conn.cursor() as cursor:
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (DB_NAME,))
        exists = cursor.fetchone()

        if not exists:
            cursor.execute(f'CREATE DATABASE "{DB_NAME}";')
            print(f"Database {DB_NAME} created.")
        else:
            print(f"Database {DB_NAME} already exists.")

    conn.close()

create_database()

def get_conn():
    data_src = os.getenv("DATABASE_URL")
    return psycopg2.connect(
        data_src
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
    hash_pass, salt = hash_password(user.password)
    with get_conn() as conn, conn.cursor() as cursor:
        cursor.execute(
            "INSERT INTO users (username, password, tokens, salt, role) VALUES (%s, %s, 0, %s, 'user')",
            (user.username, hash_pass, salt)
        )
        conn.commit()
    return user


def get_users():
    with get_conn() as conn, conn.cursor() as cursor:
        cursor.execute("SELECT username, tokens FROM users")
        rows = cursor.fetchall()
        return [UserOut(username=row[0], tokens=row[1]) for row in rows]


def get_user_by_username(username: str):
    with get_conn() as conn, conn.cursor() as cursor:
        cursor.execute("SELECT username, tokens FROM users WHERE username = %s", (username,))
        row = cursor.fetchone()
        if row:
            return UserOut(username=row[0], tokens=row[1])
        else:
            return None


def get_user_auth(username: str):
    with get_conn() as conn, conn.cursor() as cursor:
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        row = cursor.fetchone()
        if row:
            return User(username=row[0], password=row[1], tokens=row[2], salt=row[3], role=row[4])
        else:
            return None


def update_user(username: str, user: User):
    hash_pass, salt = hash_password(user.password)
    with get_conn() as conn, conn.cursor() as cursor:
        cursor.execute("UPDATE users SET password = %s, tokens = %s, salt = %s WHERE username = %s",
                       (hash_pass, user.tokens, salt, username))
        conn.commit()
        return cursor.rowcount == 1


# possible password update
"""
def update_password(username: str, new_password: str):
    hash_pass, salt = hash_password(new_password)
    with get_conn() as conn, conn.cursor() as cursor:
         cursor.execute("UPDATE users SET password = %s, salt = %s WHERE username = %s",
                        (hash_pass, salt, username))
         conn.commit()
         return cursor.rowcount == 1
"""


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


def verify_user(username: str, password: str):
    user = get_user_auth(username)
    if not user:
        return None
    if not verify_password(password, user.password, user.salt):
        return None
    return {'username': user.username, 'tokens': user.tokens, 'role': user.role}
