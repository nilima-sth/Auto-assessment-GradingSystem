import sqlite3
from pathlib import Path

from backend.app.core.config import get_settings


def _database_path(db_path: str | Path | None = None) -> Path:
    return Path(db_path).resolve() if db_path else get_settings().resolved_database_path


def init_db(db_path: str | Path | None = None) -> None:
    conn = sqlite3.connect(_database_path(db_path))
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            uploaded_pdf TEXT
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            question_number TEXT,
            raw_ocr TEXT,
            mapped_answer TEXT,
            FOREIGN KEY (student_id) REFERENCES students(id)
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS grades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            answer_id INTEGER,
            marks REAL,
            feedback TEXT,
            FOREIGN KEY (answer_id) REFERENCES answers(id)
        );
        """
    )
    conn.commit()
    conn.close()


def insert_student(name: str, pdf_name: str, db_path: str | Path | None = None) -> int:
    conn = sqlite3.connect(_database_path(db_path))
    cur = conn.cursor()
    cur.execute("INSERT INTO students (name, uploaded_pdf) VALUES (?,?)", (name, pdf_name))
    student_id = cur.lastrowid
    conn.commit()
    conn.close()
    return int(student_id)


def insert_answer(
    student_id: int,
    question_number: str,
    raw_ocr: str,
    mapped_answer: str,
    db_path: str | Path | None = None,
) -> int:
    conn = sqlite3.connect(_database_path(db_path))
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO answers (student_id, question_number, raw_ocr, mapped_answer) VALUES (?,?,?,?)",
        (student_id, question_number, raw_ocr, mapped_answer),
    )
    answer_id = cur.lastrowid
    conn.commit()
    conn.close()
    return int(answer_id)


def insert_grade(
    answer_id: int,
    marks: float,
    feedback: str,
    db_path: str | Path | None = None,
) -> None:
    conn = sqlite3.connect(_database_path(db_path))
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO grades (answer_id, marks, feedback) VALUES (?,?,?)",
        (answer_id, marks, feedback),
    )
    conn.commit()
    conn.close()
