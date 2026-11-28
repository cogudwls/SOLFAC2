from typing import Literal
from pydantic import BaseModel


# 운동 영상 정보 응답
class VideoResponse(BaseModel):
    video_id: str
    session_id: str
    file_path: str
    started_at: str
    duration_sec: float | None = None
    deleted_at: str | None = None

    deletion_status: Literal[
        "active",
        "deleted",
        "error"
    ]