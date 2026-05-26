import psycopg2

class DBAdmin:
    def __init__(self, db_config):
        self.db_config= db_config.copy()
        self.conn = psycopg2.connect(host=self.db_config['DB_HOST'],
                                database=self.db_config['DB_NAME'],
                                user=self.db_config['DB_USER'],
                                password=self.db_config['DB_PASSWORD'],
                                port=self.db_config['DB_PORT'])
        self.cur = self.conn.cursor()

    def __del__(self):
        self.disconnect()

    def reconnect(self):
        self.__init__(self.db_config)

    def disconnect(self):
        if self.conn:
            self.conn.rollback()
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()

    def get_all_persons(self):
        self.cur.execute('''
            SELECT *
            FROM persons;
        ''')
        return self.cur.fetchall()

    def get_all_records(self):
        self.cur.execute('''
            SELECT *
            FROM records;
        ''')
        return self.cur.fetchall()

    def get_person(self, person_id):
        self.cur.execute(f'''
            SELECT *
            FROM persons
            WHERE person_id = {person_id};
        ''')
        return self.cur.fetchone()

    def get_record(self, record_id):
        self.cur.execute(f'''
            SELECT *
            FROM records
            WHERE record_id = {record_id};
        ''')
        return self.cur.fetchone()

    def add_person(self, first_name, last_name, surname, birthdate):
        self.cur.execute(f'''
            INSERT INTO persons (first_name, last_name, surname, birthdate)
            VALUES
            ('{first_name}', '{last_name}', '{surname}', '{birthdate}')
            RETURNING person_id;;
        ''')
        self.conn.commit()
        return self.cur.fetchone()[0]

    def add_record(self, person_id, embedding):
        self.cur.execute(f'''
            INSERT INTO records (person_id, embedding)
            VALUES
            ({person_id}, '{embedding}')
            RETURNING record_id;;
        ''')
        self.conn.commit()
        return self.cur.fetchone()[0]

    def delete_person(self, person_id, ensure):
        if ensure:
            self.cur.execute(f'''
                DELETE FROM persons
                WHERE person_id = {person_id};
            ''')
            self.conn.commit()

    def delete_record(self, record_id, ensure):
        if ensure:
            self.cur.execute(f'''
                        DELETE FROM records
                        WHERE record_id = {record_id};
                    ''')
            self.conn.commit()

    def alter_person(self, person_id, new_first_name, new_last_name, new_surname, new_birthdate):
        self.cur.execute(f'''
            UPDATE persons
            SET first_name = '{new_first_name}',
                last_name = '{new_last_name}',
                surname = '{new_surname}',
                birthdate = '{new_birthdate}'
            WHERE person_id = {person_id};
        ''')
        self.conn.commit()

    def alter_record(self, record_id, new_person_id, new_embedding):
        self.cur.execute(f'''
            UPDATE records
            SET person_id = '{new_person_id}',
                embedding = {new_embedding}
            WHERE record_id = {record_id};
        ''')
        self.conn.commit()

    def truncate_persons(self, ensure):
        if ensure:
            self.cur.execute(f'''
                TRUNCATE TABLE persons RESTART IDENTITY CASCADE;
            ''')
            self.conn.commit()

    def truncate_records(self, ensure):
        if ensure:
            self.cur.execute(f'''
                TRUNCATE TABLE records RESTART IDENTITY;
            ''')
            self.conn.commit()

    def truncate_db(self, ensure):
        self.truncate_records(ensure)
        self.truncate_persons(ensure)

    def exec_query(self, query, password):
        if password == self.db_config['DB_ADMIN_PASSWORD']:
            self.cur.execute(query)
            self.conn.commit()
            return self.cur.fetchall()
        return '<WRONG_PASSWORD!>'