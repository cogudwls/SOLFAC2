from pydantic import BaseModel, Field


# 운동 세션 시작 요청
class SessionCreate(BaseModel):
    user_id: str
    exercise_id: str


# 운동 종료 요청 데이터
class SessionFinish(BaseModel):
    rep_count: int = Field(ge=0)
    correct_rep_count: int = Field(ge=0)