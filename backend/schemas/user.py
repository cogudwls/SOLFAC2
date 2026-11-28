from pydantic import BaseModel, Field


# 사용자 등록 요청 데이터
class UserCreate(BaseModel):
    nickname: str = Field(min_length=1, max_length=50)
    consent_yn: bool = False


# 영상 저장 동의 변경 요청
class ConsentUpdate(BaseModel):
    consent_yn: bool