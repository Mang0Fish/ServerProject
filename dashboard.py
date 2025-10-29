import streamlit as st
import psycopg2
import os
from dotenv import load_dotenv

"""
To run streamlit

streamlit run dashboard.py
"""


# Load env vars
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# Page title
st.title("📊 User Tokens Dashboard")


# DB Query
def fetch_users():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("SELECT username, tokens FROM users ORDER BY tokens DESC")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        st.error(f"Database error: {e}")
        return []


# Display table
data = fetch_users()
if data:
    st.subheader("User Token Table")
    st.dataframe({"Username": [r[0] for r in data], "Tokens": [r[1] for r in data]})
else:
    st.warning("No user data found.")
