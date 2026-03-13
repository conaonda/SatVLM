"""
Satellite VLM API Server
위성영상 분석을 위한 로컬 REST API 서버
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.routers import describe, compare, cloud, segment
from app.services.model_manager import ModelManager
from config.settings import settings

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작/종료 시 모델 로드/언로드"""
    logger.info("🛰️  Satellite VLM API 서버 시작 중...")
    model_manager = ModelManager()
    await model_manager.load()
    app.state.model_manager = model_manager
    logger.info(f"✅ 모델 로드 완료: {settings.MODEL_NAME}")
    yield
    logger.info("🔄 모델 언로드 중...")
    await model_manager.unload()
    logger.info("👋 서버 종료")


app = FastAPI(
    title="Satellite VLM API",
    description="위성영상 분석을 위한 Vision-Language Model REST API",
    version=__version__,
    lifespan=lifespan,
)

# CORS (내부망 전용이므로 전체 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# 요청 시간 로깅 미들웨어
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({elapsed:.2f}s)")
    return response


# 라우터 등록
app.include_router(describe.router, prefix="/api/v1", tags=["영상 설명"])
app.include_router(compare.router, prefix="/api/v1", tags=["영상 비교"])
app.include_router(cloud.router, prefix="/api/v1", tags=["구름 분석"])
app.include_router(segment.router, prefix="/api/v1", tags=["세그멘테이션/인지"])


@app.get("/health", tags=["시스템"])
async def health_check(request: Request):
    """서버 상태 및 모델 정보 확인"""
    manager = request.app.state.model_manager
    return {
        "status": "ok",
        "model": settings.MODEL_NAME,
        "model_loaded": manager.is_loaded,
        "device": settings.DEVICE,
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"처리되지 않은 예외: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "내부 서버 오류", "detail": str(exc)},
    )
