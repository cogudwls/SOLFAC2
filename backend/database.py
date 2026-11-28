import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "solfac.db"
SCHEMA_PATH = BASE_DIR / "schema.sql"


# 데이터베이스 연결
def get_db_connection():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    return conn


# 데이터베이스 초기화 및 테이블 생성
def init_db():
    conn = get_db_connection()

    try:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            conn.executescript(f.read())

        conn.commit()

        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()

        print("Database initialized")

        for table in tables:
            print(f"Table: {table['name']}")

    finally:
        conn.close()


if __name__ == "__main__":
    init_db()