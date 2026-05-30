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
        self.queries = {
            'get_all_persons': '''
                                SELECT *
                                FROM persons;
                                ''',
            'get_all_records': '''
                                SELECT *
                                FROM records;
                                ''',
            'get_person': '''
                            SELECT *
                            FROM persons
                            WHERE person_id = %s;
                            ''',
            'get_record': '''
                            SELECT *
                            FROM records
                            WHERE record_id = %s;
                            ''',
            'find_person': '''
                            SELECT person_id
                            FROM persons
                            WHERE first_name = %S AND last_name = %s AND surname = %s;
                            ''',
            'add_person': '''
                            INSERT INTO persons (first_name, last_name, surname, birthdate)
                            VALUES
                            (%s, %s, %s, %s)
                            RETURNING person_id;
                            ''',
            'add_record': '''
                            INSERT INTO records (person_id, embedding)
                            VALUES
                            (%s, %s)
                            RETURNING record_id;;
                            ''',
            'delete_person': '''
                                DELETE FROM persons
                                WHERE person_id = %s;
                                ''',
            'delete_record': '''
                                DELETE FROM records
                                WHERE record_id = %s;
                                ''',
            'alter_person': '''
                            UPDATE persons
                            SET first_name = %s,
                                last_name = %s,
                                surname = %s,
                                birthdate = %s
                            WHERE person_id = %s;
                            ''',
            'alter_record': '''
                            UPDATE records
                            SET person_id = %s,
                                embedding = %s
                            WHERE record_id = %s;
                            ''',
            'truncate_persons': '''
                                TRUNCATE TABLE persons RESTART IDENTITY CASCADE;
                                ''',
            'truncate_records': '''
                                TRUNCATE TABLE records RESTART IDENTITY;
                                ''',
            'get_closest': '''
                            SELECT record_id, person_id, embedding <=> %s::vector AS distance
                            FROM records
                            WHERE embedding <=> %s::vector < %s
                            ORDER BY distance ASC;
                           '''
        }

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
        self.cur.execute(self.queries['get_all_persons'])
        return self.cur.fetchall()

    def get_all_records(self):
        self.cur.execute(self.queries['get_all_records'])
        return self.cur.fetchall()

    def get_person(self, person_id):
        self.cur.execute(self.queries['get_person'], (person_id, ))
        return self.cur.fetchone()

    def get_record(self, record_id):
        self.cur.execute(self.queries['get_record'], (record_id, ))
        return self.cur.fetchone()

    def find_person(self, first_name, last_name, surname):
        self.cur.execute(self.queries['find_person'], (first_name, last_name, surname))
        return self.cur.fetchone()

    def add_person(self, first_name, last_name, surname, birthdate):
        self.cur.execute(self.queries['add_person'], (first_name, last_name, surname, birthdate))
        self.conn.commit()
        return self.cur.fetchone()[0]

    def add_record(self, person_id, embedding):
        self.cur.execute(self.queries['add_record'], (person_id, embedding))
        self.conn.commit()
        return self.cur.fetchone()[0]

    def delete_person(self, person_id, ensure):
        if ensure:
            self.cur.execute(self.queries['delete_person'], (person_id, ))
            self.conn.commit()

    def delete_record(self, record_id, ensure):
        if ensure:
            self.cur.execute(self.queries['delete_record'], (record_id, ))
            self.conn.commit()

    def alter_persons(self, person_id, new_first_name, new_last_name, new_surname, new_birthdate):
        self.cur.execute(self.queries['alter_persons'], (new_first_name, new_last_name, new_surname, new_birthdate, person_id))
        self.conn.commit()

    def alter_record(self, record_id, new_person_id, new_embedding):
        self.cur.execute(self.queries['alter_record'], (new_person_id, new_embedding, record_id))
        self.conn.commit()

    def truncate_persons(self, ensure):
        if ensure:
            self.cur.execute(self.queries['truncate_persons'])
            self.conn.commit()

    def truncate_records(self, ensure):
        if ensure:
            self.cur.execute(self.queries['truncate_records'])
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

    def get_closest(self, embedding, threshold=0.3):
        self.cur.execute(self.queries['get_closest'], (embedding, embedding, threshold))
        return self.cur.fetchall()