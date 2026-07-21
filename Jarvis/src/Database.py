import dotenv
import logging
import os
import psycopg2

class Database:
    # CREATE TABLE users
    # (
    #     login VARCHAR(64) PRIMARY KEY,
    #     pass VARCHAR(64) NOT NULL,
    #     tok VARCHAR(2048) NOT NULL
    # );
    def __init__(self):
        dotenv.load_dotenv()
        self.conn = psycopg2.connect(
            database=os.getenv("DBNAME"),
            user=os.getenv("USER"),
            password=os.getenv("PASSWORD"),
            host=os.getenv("HOST"),
            port=os.getenv("PORT")
        )
        self.curr = self.conn.cursor()
        self.queries = {
            'check': '''SELECT 1
                        FROM users
                        WHERE login = %s AND pass = %s;''',
            'save': '''INSERT INTO users (login, pass, tok)
                        VALUES (%s, %s, %s);''',
            'get': '''SELECT tok
                        FROM users
                        WHERE login = %s AND pass = %s;''',
        }
    def check_user_in_db(self, username, password):
        self.curr.execute(self.queries['check'], (username, password))
        result = self.curr.fetchone()
        if result is None:
            logging.info(f'Undefined user: {username}')
            return False
        else:
            logging.info(f'Defined user: {username}')
            return True

    def save_user_to_db(self, username, password, token):
        self.curr.execute(self.queries['save'], (username, password, token))
        self.conn.commit()
        logging.info(f'Save user: {username}')

    def get_token(self, username, password):
        self.curr.execute(self.queries['get'], (username, password))
        result = self.curr.fetchone()
        if result is None:
            raise Exception('Get token failure')
        else:
            return result[0]

    def close(self):
        logging.info('Close database')
        self.conn.close()