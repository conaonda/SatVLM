"""
구름 커버리지 분석 엔드포인트
POST /api/v1/cloud
"""

import re
import time
import logging

from fastapi import APIRouter, UploadFile, File, Request, HTTPException

from app.models.responses import CloudAnalysisResponse
from app.utils.image_utils import (
    load_image_from_bytes, resize_for_model, validate_image_size
)
from app.utils.prompts import get_cloud_prompt
from config.settings import settings

router = APIRouter()
logger = logging.getLogger(__name__)


def _parse_cloud_coverage(text: str) -> tuple[float | None, int | None]:
    """
    모델 텍스트 응답에서 구름 커버리지(%) 및 품질 점수(1-10) 파싱
    """
    coverage = None
    quality = None

    # 전체 커버리지 추출
    patterns_coverage = [
        r"overall cloud coverage[:\s]+(\d+(?:\.\d+)?)\s*%",
        r"cloud coverage[:\s]+(\d+(?:\.\d+)?)\s*%",
        r"전체 구름[:\s]+(\d+(?:\.\d+)?)\s*%",
        r"구름\s*커버리지[:\s]+(\d+(?:\.\d+)?)\s*%",
    ]
    for pat in patterns_coverage:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            coverage = float(m.group(1))
            break

    # 품질 점수 추출
    patterns_quality = [
        r"quality score[:\s]+\[?(\d+)[/\]10]",
        r"quality score[:\s]+(\d+)",
        r"품질 점수[:\s]+\[?(\d+)[/\]10]",
        r"품질 점수[:\s]+(\d+)",
    ]
    for pat in patterns_quality:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            quality = min(10, max(1, int(m.group(1))))
            break

    return coverage, quality


@router.post(
    "/cloud",
    response_model=CloudAnalysisResponse,
    summary="구름 커버리지 분석",
    description="위성영상의 구름 정도를 분석합니다. 두꺼운 구름/얇은 구름 구분, 공간 분포, 품질 점수 제공.",
)
async def analyze_cloud(
    request: Request,
    image: UploadFile = File(..., description="분석할 위성영상"),
):
    """
    구름 분석 API

    분석 결과:
    - 전체/두꺼운/얇은 구름 커버리지 비율
    - 공간 분포 (어느 영역에 구름이 집중)
    - 구름 유형 (적운, 층운, 권운 등)
    - 활용 가능성 평가 (Excellent ~ Unusable)
    - 품질 점수 (1-10)

    **예시 요청:**
    ```bash
    curl -X POST http://localhost:8000/api/v1/cloud \\
      -F "image=@sentinel2.tif"
    ```
    """
    start = time.time()

    data = await image.read()
    try:
        validate_image_size(data)
    except ValueError as e:
        raise HTTPException(status_code=413, detail=str(e))

    try:
        img = load_image_from_bytes(data, image.filename or "")
        img = resize_for_model(img)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"이미지 처리 실패: {e}")

    prompt = get_cloud_prompt()

    manager = request.app.state.model_manager
    try:
        result = manager.infer(prompt=prompt, images=[img])
    except Exception as e:
        logger.error(f"구름 분석 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"모델 추론 실패: {e}")

    # 수치 파싱 시도
    coverage, quality = _parse_cloud_coverage(result)

    elapsed_ms = (time.time() - start) * 1000
    logger.info(
        f"cloud 분석 완료: {elapsed_ms:.0f}ms, "
        f"coverage={coverage}%, quality={quality}"
    )

    return CloudAnalysisResponse(
        analysis=result,
        cloud_coverage_estimate=coverage,
        quality_score=quality,
        model=settings.MODEL_NAME,
        processing_time_ms=round(elapsed_ms, 1),
    )
