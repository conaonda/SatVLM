"""
단일 위성영상 설명 엔드포인트
POST /api/v1/describe
"""

import time
import logging

from fastapi import APIRouter, UploadFile, File, Form, Request, HTTPException
from typing import Optional

from app.models.responses import DescribeResponse
from app.utils.image_utils import (
    load_image_from_bytes, resize_for_model, validate_image_size
)
from app.utils.prompts import get_describe_prompt
from config.settings import settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/describe",
    response_model=DescribeResponse,
    summary="위성영상 설명",
    description="단일 위성영상을 분석하여 상세 설명을 반환합니다. GeoTIFF 포함 지원.",
)
async def describe_image(
    request: Request,
    image: UploadFile = File(..., description="분석할 위성영상 (JPEG/PNG/TIFF/GeoTIFF)"),
    context: Optional[str] = Form(
        None,
        description="영상 컨텍스트 (예: 'post-flood disaster area in Seoul, 2024')"
    ),
    language: Optional[str] = Form(None, description="응답 언어 (ko/en, 미입력시 설정값 사용)"),
):
    """
    위성영상 설명 API

    - **image**: 위성영상 파일 (JPEG, PNG, TIFF, GeoTIFF)
    - **context**: 선택적 컨텍스트 (지역, 이벤트 등을 알면 정확도 향상)
    - **language**: 응답 언어 (ko=한국어, en=영어)

    **예시 요청:**
    ```bash
    curl -X POST http://localhost:8000/api/v1/describe \\
      -F "image=@satellite.tif" \\
      -F "context=post-typhoon Busan port area" \\
      -F "language=ko"
    ```
    """
    start = time.time()

    # 파일 읽기 및 검증
    data = await image.read()
    try:
        validate_image_size(data)
    except ValueError as e:
        raise HTTPException(status_code=413, detail=str(e))

    # 이미지 로드 (GeoTIFF 포함)
    try:
        img = load_image_from_bytes(data, image.filename or "")
        img = resize_for_model(img)
    except Exception as e:
        logger.error(f"이미지 로드 실패: {e}")
        raise HTTPException(status_code=422, detail=f"이미지 처리 실패: {e}")

    # 언어 오버라이드
    if language:
        import config.settings as cfg_module
        orig_lang = cfg_module.settings.DEFAULT_LANGUAGE
        cfg_module.settings.DEFAULT_LANGUAGE = language

    # 프롬프트 생성
    prompt = get_describe_prompt(context=context)

    # 모델 추론
    manager = request.app.state.model_manager
    try:
        result = manager.infer(prompt=prompt, images=[img])
    except Exception as e:
        logger.error(f"모델 추론 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"모델 추론 실패: {e}")
    finally:
        if language:
            cfg_module.settings.DEFAULT_LANGUAGE = orig_lang

    elapsed_ms = (time.time() - start) * 1000
    logger.info(f"describe 완료: {elapsed_ms:.0f}ms, image={image.filename}")

    return DescribeResponse(
        description=result,
        model=settings.MODEL_NAME,
        image_size=img.size,
        processing_time_ms=round(elapsed_ms, 1),
    )
