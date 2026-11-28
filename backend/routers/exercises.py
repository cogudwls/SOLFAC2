from fastapi import APIRouter
from database import get_db_connection

router = APIRouter(
    prefix="/api/exercises",
    tags=["운동 종목"]
)


# 운동 종목 목록 조회
@router.get("")
def get_exercises():
    conn = get_db_connection()

    try:
        rows = conn.execute(
            "SELECT * FROM exercises"
        ).fetchall()

        return [dict(row) for row in rows]

    finally:
        conn.close()