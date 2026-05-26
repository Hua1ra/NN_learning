CREATE EXTENSION IF NOT EXISTS vector;

CREATE DATABASE face_recognition;

CREATE TABLE persons
(
	person_id SERIAL PRIMARY KEY,
	first_name VARCHAR(32) NOT NULL,
	last_name VARCHAR(32) NOT NULL,
	surname VARCHAR(32) NOT NULL,
	birthdate DATE NOT NULL
);

CREATE TABLE records
(
	record_id SERIAL PRIMARY KEY,
	person_id INT NOT NULL,
	embedding VECTOR(512)
);

ALTER TABLE records
ADD CONSTRAINT records_person_fk FOREIGN KEY (person_id) REFERENCES persons(person_id) ON DELETE CASCADE;