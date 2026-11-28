from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from routers import exercises, users, sessions, results, posture, videos
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="SOLFAC2 API",
    description="AI 기반 운동 성과 측정 및 자세 교정 시스템",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 기능별 API 라우터 연결
app.include_router(exercises.router)
app.include_router(users.router)
app.include_router(sessions.router)
app.include_router(results.router)
app.include_router(posture.router)
app.include_router(videos.router)


# 서버 기본 접속 확인
@app.get("/", tags=["서버 상태"])
def root():
    return {
        "service": "SOLFAC2",
        "message": "Server is running"
    }


# 서버 상태 확인
@app.get("/api/health", tags=["서버 상태"])
def health_check():
    return {"status": "online"}


# 프론트엔드 파일 경로
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


# 프론트엔드 웹페이지 제공
app.mount(
    "/web",
    StaticFiles(directory=str(FRONTEND_DIR)),
    name="frontend"
)