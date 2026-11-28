from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone
from uuid import uuid4

from database import get_db_connection
from schemas.user import UserCreate, ConsentUpdate


router = APIRouter(
    prefix="/api/users",
    tags=["사용자 관리"]
)


# 익명 사용자 등록
@router.post("", status_code=201)
def create_user(user: UserCreate):
    user_id = f"USR-{uuid4().hex}"
    created_at = datetime.now(timezone.utc).isoformat()

    consent_yn = int(user.consent_yn)
    consented_at = created_at if consent_yn else None

    conn = get_db_connection()

    try:
        conn.execute(
            """
            INSERT INTO users (
                user_id,
                nickname,
                created_at,
                consent_yn,
                consented_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                user_id,
                user.nickname,
                created_at,
                consent_yn,
                consented_at
            )
        )

        conn.commit()

        return {
            "user_id": user_id,
            "nickname": user.nickname,
            "created_at": created_at,
            "consent_yn": consent_yn,
            "consented_at": consented_at
        }

    finally:
        conn.close()


# 전체 사용자 목록 조회
@router.get("")
def get_users():
    conn = get_db_connection()

    try:
        rows = conn.execute(
            "SELECT * FROM users ORDER BY created_at DESC"
        ).fetchall()

        return [dict(row) for row in rows]

    finally:
        conn.close()


# 개별 사용자 정보 조회
@router.get("/{user_id}")
def get_user(user_id: str):
    conn = get_db_connection()

    try:
        row = conn.execute(
            "SELECT * FROM users WHERE user_id = ?",
            (user_id,)
        ).fetchone()

        if row is None:
            raise HTTPException(
                status_code=404,
                detail="사용자를 찾을 수 없습니다."
            )

        return dict(row)

    finally:
        conn.close()


# 영상 저장 동의 여부 변경
@router.patch("/{user_id}/consent")
def update_consent(user_id: str, consent: ConsentUpdate):
    conn = get_db_connection()

    try:
        user = conn.execute(
            "SELECT * FROM users WHERE user_id = ?",
            (user_id,)
        ).fetchone()

        if user is None:
            raise HTTPException(
                status_code=404,
                detail="사용자를 찾을 수 없습니다."
            )

        consent_yn = int(consent.consent_yn)

        if consent_yn:
            consented_at = (
                user["consented_at"]
                if user["consent_yn"] == 1
                else datetime.now(timezone.utc).isoformat()
            )
        else:
            consented_at = None

        conn.execute(
            """
            UPDATE users
            SET consent_yn = ?, consented_at = ?
            WHERE user_id = ?
            """,
            (consent_yn, consented_at, user_id)
        )

        conn.commit()

        return {
            "user_id": user_id,
            "consent_yn": consent_yn,
            "consented_at": consented_at
        }

    finally:
        conn.close()