from pydantic import BaseModel
import psycopg2
from models import *


conn = psycopg2.connect(
    dbname="ServerUsers",
    user="postgres",
    password="1234",
    host="localhost",
    port="5432"
)


cursor = conn.cursor()


def create_table():
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
    cursor.execute("INSERT INTO users (username, password, tokens) VALUES (?, ?, ?)",
                   (user.username, user.password, user.tokens)
    )
    conn.commit()
    return user


cursor.close()
conn.close()
