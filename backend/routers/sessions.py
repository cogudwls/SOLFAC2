from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone
from uuid import uuid4

from database import get_db_connection
from schemas.session import SessionCreate, SessionFinish


router = APIRouter(
    prefix="/api/sessions",
    tags=["운동 세션 관리"]
)


# 운동 세션 시작
@router.post("", status_code=201)
def create_session(session: SessionCreate):
    conn = get_db_connection()

    try:
        # 사용자 존재 여부 확인
        user = conn.execute(
            "SELECT user_id FROM users WHERE user_id = ?",
            (session.user_id,)
        ).fetchone()

        if user is None:
            raise HTTPException(
                status_code=404,
                detail="사용자를 찾을 수 없습니다."
            )

        # 운동 종류 존재 여부 확인
        exercise = conn.execute(
            "SELECT exercise_id FROM exercises WHERE exercise_id = ?",
            (session.exercise_id,)
        ).fetchone()

        if exercise is None:
            raise HTTPException(
                status_code=404,
                detail="운동 종목을 찾을 수 없습니다."
            )

        # 세션 ID 및 시작 시각 생성
        session_id = f"SES-{uuid4().hex}"
        started_at = datetime.now(timezone.utc).isoformat()

        # 운동 세션 저장
        conn.execute(
            """
            INSERT INTO sessions (
                session_id,
                user_id,
                exercise_id,
                started_at,
                ended_at,
                status
            )
            VALUES (?, ?, ?, ?, NULL, 'running')
            """,
            (
                session_id,
                session.user_id,
                session.exercise_id,
                started_at
            )
        )

        conn.commit()

        return {
            "session_id": session_id,
            "user_id": session.user_id,
            "exercise_id": session.exercise_id,
            "started_at": started_at,
            "ended_at": None,
            "status": "running"
        }

    finally:
        conn.close()


# 개별 운동 세션 조회
@router.get("/{session_id}")
def get_session(session_id: str):
    conn = get_db_connection()

    try:
        row = conn.execute(
            """
            SELECT *
            FROM sessions
            WHERE session_id = ?
            """,
            (session_id,)
        ).fetchone()

        if row is None:
            raise HTTPException(
                status_code=404,
                detail="운동 세션을 찾을 수 없습니다."
            )

        return dict(row)

    finally:
        conn.close()


# 운동 종료 및 결과 저장
@router.post("/{session_id}/finish")
def finish_session(session_id: str, result: SessionFinish):

    if result.correct_rep_count > result.rep_count:
        raise HTTPException(
            status_code=400,
            detail="정자세 횟수는 전체 운동 횟수를 초과할 수 없습니다."
        )

    conn = get_db_connection()

    try:
        conn.execute("BEGIN IMMEDIATE")

        # 세션 존재 여부 확인
        session = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,)
        ).fetchone()

        if session is None:
            raise HTTPException(
                status_code=404,
                detail="운동 세션을 찾을 수 없습니다."
            )

        # 진행 중인 세션만 종료 가능
        if session["status"] != "running":
            raise HTTPException(
                status_code=409,
                detail="진행 중인 운동 세션이 아닙니다."
            )

        # 종료 시각 및 운동 시간 계산
        ended_at = datetime.now(timezone.utc)

        started_at = datetime.fromisoformat(
            session["started_at"]
        )

        duration_sec = round(
            (ended_at - started_at).total_seconds(), 2
        )

        result_id = f"RES-{uuid4().hex}"

        # 세션 상태 변경
        conn.execute(
            """
            UPDATE sessions
            SET ended_at = ?, status = 'completed'
            WHERE session_id = ?
            """,
            (ended_at.isoformat(), session_id)
        )

        # 운동 결과 저장
        conn.execute(
            """
            INSERT INTO exercise_results (
                result_id,
                session_id,
                rep_count,
                correct_rep_count,
                duration_sec,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                result_id,
                session_id,
                result.rep_count,
                result.correct_rep_count,
                duration_sec,
                ended_at.isoformat()
            )
        )

        conn.commit()

        return {
            "result_id": result_id,
            "session_id": session_id,
            "status": "completed",
            "rep_count": result.rep_count,
            "correct_rep_count": result.correct_rep_count,
            "duration_sec": duration_sec,
            "ended_at": ended_at.isoformat()
        }

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()