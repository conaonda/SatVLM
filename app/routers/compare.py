"""
위성영상 비교 엔드포인트
POST /api/v1/compare
"""

import time
import logging

from fastapi import APIRouter, UploadFile, File, Form, Request, HTTPException
from typing import Optional

from app.models.responses import CompareResponse
from app.utils.image_utils import (
    load_image_from_bytes, resize_for_model, validate_image_size
)
from app.utils.prompts import get_compare_prompt
from config.settings import settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/compare",
    response_model=CompareResponse,
    summary="위성영상 비교",
    description="두 위성영상을 비교 분석합니다. 시계열 변화 감지 또는 일반 비교 모드 지원.",
)
async def compare_images(
    request: Request,
    image1: UploadFile = File(..., description="첫 번째 영상 (이전/기준 영상)"),
    image2: UploadFile = File(..., description="두 번째 영상 (이후/비교 영상)"),
    mode: str = Form(
        "temporal",
        description="비교 모드: 'temporal' (시계열 변화) | 'general' (일반 비교)"
    ),
    context: Optional[str] = Form(
        None,
        description="비교 컨텍스트 (예: '2023년 홍수 전후 부산항')"
    ),
):
    """
    위성영상 비교 API

    - **image1**: 첫 번째 영상 (시계열 비교 시: 이전 영상)
    - **image2**: 두 번째 영상 (시계열 비교 시: 이후 영상)
    - **mode**: `temporal` (시계열 변화 감지) | `general` (일반 내용 비교)
    - **context**: 선택적 맥락 정보

    **예시 요청:**
    ```bash
    curl -X POST http://localhost:8000/api/v1/compare \\
      -F "image1=@before.tif" \\
      -F "image2=@after.tif" \\
      -F "mode=temporal" \\
      -F "context=태풍 힌남노 전후 포항제철소"
    ```
    """
    if mode not in ("temporal", "general"):
        raise HTTPException(
            status_code=422,
            detail="mode는 'temporal' 또는 'general' 이어야 합니다."
        )

    start = time.time()

    # 두 이미지 로드
    images = []
    sizes = []
    for i, upload in enumerate([image1, image2], 1):
        data = await upload.read()
        try:
            validate_image_size(data)
        except ValueError as e:
            raise HTTPException(status_code=413, detail=f"image{i}: {e}")

        try:
            img = load_image_from_bytes(data, upload.filename or "")
            img = resize_for_model(img)
            images.append(img)
            sizes.append(img.size)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"image{i} 처리 실패: {e}")

    # 프롬프트 생성
    prompt = get_compare_prompt(mode=mode)
    if context:
        prompt = f"Context: {context}\n\n" + prompt

    # 모델 추론
    manager = request.app.state.model_manager
    try:
        result = manager.infer(prompt=prompt, images=images)
    except Exception as e:
        logger.error(f"비교 추론 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"모델 추론 실패: {e}")

    elapsed_ms = (time.time() - start) * 1000
    logger.info(f"compare 완료: {elapsed_ms:.0f}ms, mode={mode}")

    return CompareResponse(
        comparison=result,
        model=settings.MODEL_NAME,
        image_sizes=sizes,
        mode=mode,
        processing_time_ms=round(elapsed_ms, 1),
    )
