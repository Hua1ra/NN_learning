import os
import dotenv
import logging
from pathlib import Path
import psycopg2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load .env.back parameters
dotenv.load_dotenv(Path(__file__).resolve().parent / '.env.back')

# Application example
app = FastAPI(title="Jarvis API")

class Database:
    def __init__(self):
        # Connect to a db
        self.conn = psycopg2.connect(
            database=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            host=os.getenv("POSTGRES_HOST"),
            port=os.getenv("POSTGRES_PORT")
        )
        self.conn.autocommit = True
        self.cur = self.conn.cursor()
        self.cur.execute('''CREATE TABLE IF NOT EXISTS users
                            (
                               login VARCHAR(64) PRIMARY KEY,
                               pass VARCHAR(64) NOT NULL,
                               tok VARCHAR(2048) NOT NULL
                            )''')
        # All possible commands
        self.queries = {
            'password': '''SELECT 1 FROM users WHERE login = %s AND pass = %s;''',
            'save': '''INSERT INTO users (login, pass, tok) VALUES (%s, %s, %s) 
                       ON CONFLICT (login) DO UPDATE SET pass = EXCLUDED.pass, tok = EXCLUDED.tok;''', # Добавил ON CONFLICT на случай перезаписи
            'get': '''SELECT tok FROM users WHERE login = %s AND pass = %s;''',
            'login': '''SELECT 1 FROM users WHERE login = %s;''',
        }

    # True if (user, password) exists in db else False
    def check_user(self, username, password):
        self.cur.execute(self.queries['password'], (username, password))
        return self.cur.fetchone() is not None

    # True if (user) exists in db else False
    def check_login(self, username):
        self.cur.execute(self.queries['login'], (username, ))
        return self.cur.fetchone() is not None

    # Save (user, password, token) to db
    def save_user(self, username, password, token):
        self.cur.execute(self.queries['save'], (username, password, token))

    # Get token for (user, password)
    def get_token(self, username, password):
        self.cur.execute(self.queries['get'], (username, password))
        result = self.cur.fetchone()
        return result[0] if result else None

# Db example for an API
db = Database()

# Data Models
class UserAuth(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str

class UserSave(BaseModel):
    username: str
    password: str
    token: str

# Corresponding endpoints for a db
@app.post("/auth/check")
def check_user_endpoint(user: UserAuth):
    exists = db.check_user(user.username, user.password)
    return {"exists": exists}

@app.post("/auth/login")
def check_login_endpoint(user: UserLogin):
    exists = db.check_login(user.username)
    return {"exists": exists}

@app.post("/auth/save")
def save_user_endpoint(user: UserSave):
    try:
        db.save_user(user.username, user.password, user.token)
        return {"status": "success"}
    except Exception as e:
        logging.error(f"Error saving user: {e}")
        raise HTTPException(detail='Error saving user', status_code=500)

@app.post("/auth/token")
def get_token_endpoint(user: UserAuth):
    token = db.get_token(user.username, user.password)
    if token is None:
        logging.error('Token error')
        raise HTTPException(detail='Token error', status_code=500)
    return {"token": token}