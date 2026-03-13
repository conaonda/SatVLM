"""
세그멘테이션 / 특정 지점 분석 엔드포인트

POST /api/v1/segment          - 전체 영상 세그멘테이션
POST /api/v1/segment/point    - 픽셀 좌표 기반 지점 설명
POST /api/v1/segment/region   - 바운딩박스 기반 영역 설명
POST /api/v1/segment/detect   - 특정 객체 감지
POST /api/v1/segment/change   - 두 영상의 특정 지점 변화 감지
"""

import time
import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, Request, HTTPException
from PIL import ImageDraw

from app.models.responses import (
    SegmentationResponse, PointDescriptionResponse, ObjectDetectionResponse
)
from app.utils.image_utils import (
    load_image_from_bytes, resize_for_model, validate_image_size, crop_region
)
from app.utils.prompts import (
    get_segmentation_prompt, get_point_prompt,
    get_object_detection_prompt, get_change_point_prompt
)
from config.settings import settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/segment",
    response_model=SegmentationResponse,
    summary="전체 영상 세그멘테이션",
    description="위성영상 전체의 토지 피복 분류 및 세그멘테이션 분석",
)
async def segment_image(
    request: Request,
    image: UploadFile = File(..., description="분석할 위성영상"),
):
    """
    **토지피복 분류:**
    도시/농경지/산림/수계/나지/혼합 등 카테고리별 비율 및 공간 분포 분석

    **예시:**
    ```bash
    curl -X POST http://localhost:8000/api/v1/segment -F "image=@scene.tif"
    ```
    """
    start = time.time()
    data = await image.read()
    validate_image_size(data)
    img = load_image_from_bytes(data, image.filename or "")
    img = resize_for_model(img)

    manager = request.app.state.model_manager
    try:
        result = manager.infer(prompt=get_segmentation_prompt(), images=[img])
    except Exception as e:
        logger.error(f"세그멘테이션 추론 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"모델 추론 실패: {e}")

    return SegmentationResponse(
        segmentation=result,
        model=settings.MODEL_NAME,
        processing_time_ms=round((time.time() - start) * 1000, 1),
    )


@router.post(
    "/segment/point",
    response_model=PointDescriptionResponse,
    summary="픽셀 좌표 기반 지점 설명",
    description="이미지 내 특정 픽셀 좌표 주변의 지물을 설명합니다.",
)
async def describe_point(
    request: Request,
    image: UploadFile = File(..., description="분석할 위성영상"),
    x: int = Form(..., description="픽셀 X 좌표 (0 = 왼쪽)"),
    y: int = Form(..., description="픽셀 Y 좌표 (0 = 위쪽)"),
    radius: int = Form(50, description="분석 반경 (픽셀, 기본값 50)"),
):
    """
    특정 픽셀 좌표를 중심으로 한 영역 설명

    **예시:**
    ```bash
    curl -X POST http://localhost:8000/api/v1/segment/point \\
      -F "image=@scene.tif" -F "x=512" -F "y=384" -F "radius=100"
    ```
    """
    start = time.time()
    data = await image.read()
    validate_image_size(data)
    img = load_image_from_bytes(data, image.filename or "")
    orig_w, orig_h = img.size

    # 좌표 유효성 검사
    if not (0 <= x < orig_w and 0 <= y < orig_h):
        raise HTTPException(
            status_code=422,
            detail=f"좌표 ({x}, {y})가 이미지 범위 ({orig_w}x{orig_h}) 밖입니다."
        )

    # 관심 영역 크롭 후 빨간 마커 표시
    bbox = {
        "x1": max(0, x - radius),
        "y1": max(0, y - radius),
        "x2": min(orig_w, x + radius),
        "y2": min(orig_h, y + radius),
    }
    cropped = crop_region(img, bbox)

    # 전체 이미지에 마커 표시 (컨텍스트 제공용)
    marked = img.copy()
    draw = ImageDraw.Draw(marked)
    draw.rectangle(
        [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]],
        outline="red", width=max(2, orig_w // 200)
    )
    marked = resize_for_model(marked)
    cropped = resize_for_model(cropped)

    prompt = get_point_prompt(x, y, orig_w, orig_h)
    manager = request.app.state.model_manager

    # 전체 이미지 + 크롭 영역 동시 제공 (컨텍스트 + 디테일)
    try:
        result = manager.infer(prompt=prompt, images=[marked, cropped])
    except Exception as e:
        logger.error(f"포인트 분석 추론 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"모델 추론 실패: {e}")

    return PointDescriptionResponse(
        description=result,
        point={"x": x, "y": y, "radius": radius, "bbox": bbox},
        model=settings.MODEL_NAME,
        processing_time_ms=round((time.time() - start) * 1000, 1),
    )


@router.post(
    "/segment/region",
    response_model=PointDescriptionResponse,
    summary="바운딩박스 영역 설명",
    description="지정한 직사각형 영역 내 지물을 상세 설명합니다.",
)
async def describe_region(
    request: Request,
    image: UploadFile = File(..., description="분석할 위성영상"),
    x1: int = Form(..., description="좌상단 X"),
    y1: int = Form(..., description="좌상단 Y"),
    x2: int = Form(..., description="우하단 X"),
    y2: int = Form(..., description="우하단 Y"),
):
    """
    직사각형 영역 지정 설명

    **예시:**
    ```bash
    curl -X POST http://localhost:8000/api/v1/segment/region \\
      -F "image=@scene.tif" -F "x1=100" -F "y1=200" -F "x2=400" -F "y2=500"
    ```
    """
    start = time.time()
    data = await image.read()
    validate_image_size(data)
    img = load_image_from_bytes(data, image.filename or "")

    if x1 >= x2 or y1 >= y2:
        raise HTTPException(status_code=422, detail="x1 < x2, y1 < y2 이어야 합니다.")

    bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    cropped = crop_region(img, bbox)

    # 원본에 박스 표시
    marked = img.copy()
    draw = ImageDraw.Draw(marked)
    draw.rectangle([x1, y1, x2, y2], outline="red", width=max(2, img.size[0] // 200))
    marked = resize_for_model(marked)
    cropped = resize_for_model(cropped)

    from app.utils.prompts import POINT_DESCRIPTION, _lang_suffix
    prompt = POINT_DESCRIPTION.format(lang=_lang_suffix())

    manager = request.app.state.model_manager
    try:
        result = manager.infer(prompt=prompt, images=[marked, cropped])
    except Exception as e:
        logger.error(f"영역 분석 추론 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"모델 추론 실패: {e}")

    return PointDescriptionResponse(
        description=result,
        point={"bbox": bbox},
        model=settings.MODEL_NAME,
        processing_time_ms=round((time.time() - start) * 1000, 1),
    )


@router.post(
    "/segment/detect",
    response_model=ObjectDetectionResponse,
    summary="특정 객체 감지",
    description="위성영상에서 특정 객체(건물, 차량, 선박, 활주로 등)를 감지하고 설명합니다.",
)
async def detect_objects(
    request: Request,
    image: UploadFile = File(..., description="분석할 위성영상"),
    target: str = Form(
        ...,
        description="감지 대상 (예: 'ships', 'aircraft', 'buildings', 'vehicles', '선박', '건물')"
    ),
):
    """
    객체 감지 및 위치 설명

    **지원 객체 (예시):**
    - ships / 선박
    - aircraft / 항공기
    - buildings / 건물
    - vehicles / 차량
    - military installations / 군사시설
    - agricultural fields / 농경지

    **예시:**
    ```bash
    curl -X POST http://localhost:8000/api/v1/segment/detect \\
      -F "image=@port.tif" -F "target=ships"
    ```
    """
    start = time.time()
    data = await image.read()
    validate_image_size(data)
    img = load_image_from_bytes(data, image.filename or "")
    img = resize_for_model(img)

    prompt = get_object_detection_prompt(target)
    manager = request.app.state.model_manager
    try:
        result = manager.infer(prompt=prompt, images=[img])
    except Exception as e:
        logger.error(f"객체 감지 추론 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"모델 추론 실패: {e}")

    return ObjectDetectionResponse(
        detections=result,
        target_object=target,
        model=settings.MODEL_NAME,
        processing_time_ms=round((time.time() - start) * 1000, 1),
    )


@router.post(
    "/segment/change",
    response_model=PointDescriptionResponse,
    summary="두 영상의 특정 지점 변화 감지",
    description="두 위성영상에서 동일 지점의 변화를 감지하고 설명합니다.",
)
async def detect_change_at_point(
    request: Request,
    image1: UploadFile = File(..., description="이전 영상"),
    image2: UploadFile = File(..., description="이후 영상"),
    x: int = Form(..., description="픽셀 X 좌표"),
    y: int = Form(..., description="픽셀 Y 좌표"),
    radius: int = Form(100, description="분석 반경 (픽셀)"),
):
    """
    특정 지점 변화 감지

    **예시:**
    ```bash
    curl -X POST http://localhost:8000/api/v1/segment/change \\
      -F "image1=@before.tif" -F "image2=@after.tif" \\
      -F "x=512" -F "y=384" -F "radius=150"
    ```
    """
    start = time.time()

    images = []
    for upload in [image1, image2]:
        data = await upload.read()
        validate_image_size(data)
        img = load_image_from_bytes(data, upload.filename or "")

        # 관심 영역 크롭
        w, h = img.size
        bbox = {
            "x1": max(0, x - radius), "y1": max(0, y - radius),
            "x2": min(w, x + radius), "y2": min(h, y + radius),
        }
        cropped = crop_region(img, bbox)

        # 마커 표시
        draw = ImageDraw.Draw(img)
        draw.rectangle(
            [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]],
            outline="red", width=max(2, w // 200)
        )
        images.append(resize_for_model(img))
        images.append(resize_for_model(cropped))

    # [전체1, 크롭1, 전체2, 크롭2] 4장 전달
    prompt = get_change_point_prompt()
    manager = request.app.state.model_manager
    try:
        result = manager.infer(prompt=prompt, images=images)
    except Exception as e:
        logger.error(f"변화 감지 추론 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"모델 추론 실패: {e}")

    return PointDescriptionResponse(
        description=result,
        point={"x": x, "y": y, "radius": radius},
        model=settings.MODEL_NAME,
        processing_time_ms=round((time.time() - start) * 1000, 1),
    )
