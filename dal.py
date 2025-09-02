from pydantic import BaseModel
import psycopg2

conn = psycopg2.connect(
    dbname="ServerUsers",
    user="postgres",
    password="1234",
    host="localhost",
    port="5432"
)


cursor = conn.cursor()
cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT NOT NULL,
                tokens INTEGER NOT NULL
            );
        """)
conn.commit()

cursor.close()
conn.close()
