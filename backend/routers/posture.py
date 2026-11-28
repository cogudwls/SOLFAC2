from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone

from database import get_db_connection
from schemas.posture import PostureCreate


router = APIRouter(
    prefix="/api/sessions",
    tags=["자세 판정 관리"]
)


# 자세 판정 기록 저장
@router.post("/{session_id}/posture", status_code=201)
def create_posture_event(
    session_id: str,
    posture: PostureCreate
):
    conn = get_db_connection()

    try:
        conn.execute("BEGIN IMMEDIATE")

        # 운동 세션 존재 여부 확인
        session = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,)
        ).fetchone()

        if session is None:
            raise HTTPException(
                status_code=404,
                detail="운동 세션을 찾을 수 없습니다."
            )

        # 진행 중인 세션인지 확인
        if session["status"] != "running":
            raise HTTPException(
                status_code=409,
                detail="진행 중인 운동 세션이 아닙니다."
            )

        detected_at = datetime.now(timezone.utc).isoformat()

        # 자세 판정 기록 저장
        cursor = conn.execute(
            """
            INSERT INTO posture_events (
                session_id,
                rep_no,
                posture_status,
                error_type,
                detected_at,
                feedback_msg
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                posture.rep_no,
                posture.posture_status,
                posture.error_type,
                detected_at,
                posture.feedback_msg
            )
        )

        event_id = cursor.lastrowid

        conn.commit()

        return {
            "event_id": event_id,
            "session_id": session_id,
            "rep_no": posture.rep_no,
            "posture_status": posture.posture_status,
            "error_type": posture.error_type,
            "detected_at": detected_at,
            "feedback_msg": posture.feedback_msg
        }

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


# 세션별 자세 판정 기록 조회
@router.get("/{session_id}/posture")
def get_posture_events(session_id: str):
    conn = get_db_connection()

    try:
        session = conn.execute(
            "SELECT session_id FROM sessions WHERE session_id = ?",
            (session_id,)
        ).fetchone()

        if session is None:
            raise HTTPException(
                status_code=404,
                detail="운동 세션을 찾을 수 없습니다."
            )

        rows = conn.execute(
            """
            SELECT *
            FROM posture_events
            WHERE session_id = ?
            ORDER BY detected_at ASC, event_id ASC
            """,
            (session_id,)
        ).fetchall()

        return [dict(row) for row in rows]

    finally:
        conn.close()