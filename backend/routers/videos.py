from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from database import get_db_connection
from schemas.video import VideoResponse


router = APIRouter(
    prefix="/api",
    tags=["운동 영상 관리"]
)

BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_DIR = BASE_DIR / "storage" / "videos"

VIDEO_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".webm", ".mov"}
MAX_FILE_SIZE = 100 * 1024 * 1024


# 운동 영상 업로드
@router.post(
    "/sessions/{session_id}/videos",
    response_model=VideoResponse,
    status_code=201
)
def upload_video(
    session_id: str,
    file: UploadFile = File(...),
    duration_sec: float | None = Form(default=None, ge=0)
):
    conn = get_db_connection()
    file_path = None
    saved = False

    try:
        # 세션 및 사용자 동의 여부 확인
        session = conn.execute(
            """
            SELECT s.*, u.consent_yn
            FROM sessions s
            JOIN users u ON s.user_id = u.user_id
            WHERE s.session_id = ?
            """,
            (session_id,)
        ).fetchone()

        if session is None:
            raise HTTPException(
                status_code=404,
                detail="운동 세션을 찾을 수 없습니다."
            )

        if not session["consent_yn"]:
            raise HTTPException(
                status_code=403,
                detail="영상 저장에 동의하지 않은 사용자입니다."
            )

        # 영상 확장자 확인
        extension = Path(file.filename or "").suffix.lower()

        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail="지원하지 않는 영상 형식입니다."
            )

        video_id = f"VID-{uuid4().hex}"

        file_path = VIDEO_DIR / f"{video_id}{extension}"

        # 실제 영상 파일 저장
        total_size = 0

        with file_path.open("xb") as output:
            while chunk := file.file.read(1024 * 1024):
                total_size += len(chunk)

                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="영상 파일은 100MB를 초과할 수 없습니다."
                    )

                output.write(chunk)

        if total_size == 0:
            raise HTTPException(
                status_code=400,
                detail="빈 영상 파일은 저장할 수 없습니다."
            )

        relative_path = file_path.relative_to(BASE_DIR).as_posix()

        # 저장 직전 동의 여부 재확인
        conn.execute("BEGIN IMMEDIATE")

        consent = conn.execute(
            """
            SELECT u.consent_yn
            FROM sessions s
            JOIN users u ON s.user_id = u.user_id
            WHERE s.session_id = ?
            """,
            (session_id,)
        ).fetchone()

        if consent is None or not consent["consent_yn"]:
            raise HTTPException(
                status_code=403,
                detail="영상 저장 동의를 확인할 수 없습니다."
            )

        # DB에 영상 정보 저장
        conn.execute(
            """
            INSERT INTO videos (
                video_id,
                session_id,
                file_path,
                started_at,
                duration_sec,
                deleted_at,
                deletion_status
            )
            VALUES (?, ?, ?, ?, ?, NULL, 'active')
            """,
            (
                video_id,
                session_id,
                relative_path,
                session["started_at"],
                duration_sec
            )
        )

        conn.commit()
        saved = True

        return {
            "video_id": video_id,
            "session_id": session_id,
            "file_path": relative_path,
            "started_at": session["started_at"],
            "duration_sec": duration_sec,
            "deleted_at": None,
            "deletion_status": "active"
        }

    except Exception:
        conn.rollback()
        raise

    finally:
        try:
            if not saved and file_path is not None:
                file_path.unlink(missing_ok=True)
        finally:
            conn.close()
            file.file.close()

# 세션별 운동 영상 조회
@router.get(
    "/sessions/{session_id}/videos",
    response_model=list[VideoResponse]
)
def get_session_videos(session_id: str):
    conn = get_db_connection()

    try:
        session = conn.execute(
            """
            SELECT session_id
            FROM sessions
            WHERE session_id = ?
            """,
            (session_id,)
        ).fetchone()

        if session is None:
            raise HTTPException(
                status_code=404,
                detail="운동 세션을 찾을 수 없습니다."
            )

        videos = conn.execute(
            """
            SELECT *
            FROM videos
            WHERE session_id = ?
            ORDER BY started_at DESC
            """,
            (session_id,)
        ).fetchall()

        return [dict(video) for video in videos]

    finally:
        conn.close()

# 운동 영상 삭제
@router.delete(
    "/videos/{video_id}",
    response_model=VideoResponse
)
def delete_video(video_id: str):
    conn = get_db_connection()

    try:
        conn.execute("BEGIN IMMEDIATE")

        video = conn.execute(
            """
            SELECT *
            FROM videos
            WHERE video_id = ?
            """,
            (video_id,)
        ).fetchone()

        if video is None:
            raise HTTPException(
                status_code=404,
                detail="영상을 찾을 수 없습니다."
            )

        # 이미 삭제된 영상 처리
        if video["deletion_status"] == "deleted":
            return dict(video)

        # 실제 파일 경로 확인
        file_path = (BASE_DIR / video["file_path"]).resolve()

        # 지정된 영상 저장 폴더 외부의 파일 삭제 방지
        if not file_path.is_relative_to(VIDEO_DIR.resolve()):
            raise HTTPException(
                status_code=400,
                detail="잘못된 영상 파일 경로입니다."
            )

        try:
            # 실제 영상 파일 삭제
            file_path.unlink(missing_ok=True)

        except OSError:
            conn.execute(
                """
                UPDATE videos
                SET deletion_status = 'error'
                WHERE video_id = ?
                """,
                (video_id,)
            )

            conn.commit()

            raise HTTPException(
                status_code=500,
                detail="영상 파일 삭제에 실패했습니다."
            )

        deleted_at = datetime.now(timezone.utc).isoformat()

        # 영상 삭제 상태 갱신
        conn.execute(
            """
            UPDATE videos
            SET deleted_at = ?,
                deletion_status = 'deleted'
            WHERE video_id = ?
            """,
            (deleted_at, video_id)
        )

        conn.commit()

        updated_video = conn.execute(
            """
            SELECT *
            FROM videos
            WHERE video_id = ?
            """,
            (video_id,)
        ).fetchone()

        return dict(updated_video)

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()