from fastapi import APIRouter, HTTPException
from database import get_db_connection


router = APIRouter(
    prefix="/api",
    tags=["운동 결과 관리"]
)


# 운동 결과 조회
@router.get("/sessions/{session_id}/result")
def get_exercise_result(session_id: str):
    conn = get_db_connection()

    try:
        row = conn.execute(
            """
            SELECT
                r.*,
                s.user_id,
                s.exercise_id,
                e.name AS exercise_name,
                s.started_at,
                s.ended_at,
                s.status
            FROM exercise_results r
            JOIN sessions s ON r.session_id = s.session_id
            JOIN exercises e ON s.exercise_id = e.exercise_id
            WHERE r.session_id = ?
            """,
            (session_id,)
        ).fetchone()

        if row is None:
            raise HTTPException(
                status_code=404,
                detail="저장된 운동 결과가 없습니다."
            )

        return dict(row)

    finally:
        conn.close()


# 사용자별 운동 이력 조회
@router.get("/users/{user_id}/history")
def get_user_history(user_id: str):
    conn = get_db_connection()

    try:
        # 사용자 존재 여부 확인
        user = conn.execute(
            "SELECT user_id FROM users WHERE user_id = ?",
            (user_id,)
        ).fetchone()

        if user is None:
            raise HTTPException(
                status_code=404,
                detail="사용자를 찾을 수 없습니다."
            )

        # 완료된 운동 기록 조회
        rows = conn.execute(
            """
            SELECT
                s.session_id,
                s.exercise_id,
                e.name AS exercise_name,
                s.started_at,
                s.ended_at,
                r.result_id,
                r.rep_count,
                r.correct_rep_count,
                r.duration_sec
            FROM sessions s
            JOIN exercises e
                ON s.exercise_id = e.exercise_id
            JOIN exercise_results r
                ON s.session_id = r.session_id
            WHERE s.user_id = ?
                AND s.status = 'completed'
            ORDER BY s.started_at DESC
            """,
            (user_id,)
        ).fetchall()

        return [dict(row) for row in rows]

    finally:
        conn.close()