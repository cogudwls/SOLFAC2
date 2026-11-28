from typing import Literal
from pydantic import BaseModel, Field


# 자세 판정 기록 요청 데이터
class PostureCreate(BaseModel):
    rep_no: int | None = Field(default=None, ge=0)

    posture_status: Literal[
        "correct",
        "incorrect",
        "undetermined"
    ]

    error_type: str | None = None
    feedback_msg: str | None = None