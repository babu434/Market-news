# FILE: auth.py
import bcrypt
import streamlit as st
import os
from sqlalchemy import create_engine, text

def get_db_engine():
    # Fix URL for SQLAlchemy
    db_url = os.environ["NEON_DB_URL"].replace("postgres://", "postgresql://")
    return create_engine(db_url)

def create_users_table():
    engine = get_db_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS users (
                    username VARCHAR(50) PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    full_name VARCHAR(100)
                );
            """))
            conn.commit()
    except Exception as e:
        st.error(f"DB Error: {e}")

def register_user(username, plain_password, full_name):
    engine = get_db_engine()
    # Check if user exists
    with engine.connect() as conn:
        result = conn.execute(text("SELECT username FROM users WHERE username = :u"), {"u": username}).fetchone()
        if result:
            return False, "User already exists!"

    # Hash password
    hashed_pw = bcrypt.hashpw(plain_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    try:
        with engine.connect() as conn:
            conn.execute(
                text("INSERT INTO users (username, password_hash, full_name) VALUES (:u, :p, :n)"),
                {"u": username, "p": hashed_pw, "n": full_name}
            )
            conn.commit()
        return True, "User created!"
    except Exception as e:
        return False, str(e)

def authenticate_user(username, plain_password):
    engine = get_db_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT password_hash, full_name FROM users WHERE username = :u"),
            {"u": username}
        ).fetchone()

    if result:
        stored_hash = result[0]
        if bcrypt.checkpw(plain_password.encode('utf-8'), stored_hash.encode('utf-8')):
            return result[1] # Return Full Name
    return None