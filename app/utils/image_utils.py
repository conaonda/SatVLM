"""
이미지 전처리 유틸리티
- JPEG/PNG/TIFF/GeoTIFF 지원
- 위성영상 특화 정규화
- 멀티밴드 → RGB 변환
"""

import io
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from config.settings import settings

logger = logging.getLogger(__name__)


def load_image_from_bytes(data: bytes, filename: str = "") -> Image.Image:
    """
    업로드된 바이트에서 PIL Image 반환
    GeoTIFF 포함 모든 위성영상 포맷 지원
    """
    ext = Path(filename).suffix.lower()

    if ext in (".tif", ".tiff") and settings.SUPPORT_GEOTIFF:
        return _load_geotiff(data)
    else:
        return _load_standard(data)


def _load_standard(data: bytes) -> Image.Image:
    """일반 이미지 (JPEG/PNG/TIFF) 로드"""
    img = Image.open(io.BytesIO(data))
    return _normalize_to_rgb(img)


def _load_geotiff(data: bytes) -> Image.Image:
    """
    GeoTIFF 로드
    - 멀티밴드 처리 (RGB, RGB+NIR, 단일 밴드 등)
    - 16비트 → 8비트 정규화
    - 지리참조 정보는 메타데이터로 추출 (픽셀 좌표 계산용)
    """
    try:
        import rasterio
        from rasterio.io import MemoryFile

        with MemoryFile(data) as memfile:
            with memfile.open() as dataset:
                count = dataset.count
                dtype = dataset.dtypes[0]
                logger.info(f"GeoTIFF: {count}밴드, dtype={dtype}, CRS={dataset.crs}")

                if count >= 3:
                    # RGB 또는 RGBNIR → RGB 첫 3밴드 사용
                    r = dataset.read(1).astype(np.float32)
                    g = dataset.read(2).astype(np.float32)
                    b = dataset.read(3).astype(np.float32)
                elif count == 1:
                    # 단일 밴드 (SAR, Panchromatic) → 그레이스케일 복제
                    band = dataset.read(1).astype(np.float32)
                    r = g = b = band
                else:
                    r = dataset.read(1).astype(np.float32)
                    g = dataset.read(2).astype(np.float32)
                    b = g.copy()

                # 16비트/32비트 → 8비트 정규화 (2-98 퍼센타일)
                def normalize_band(arr: np.ndarray) -> np.ndarray:
                    p2, p98 = np.percentile(arr[arr > 0], [2, 98]) if arr.any() else (0, 1)
                    arr = np.clip(arr, p2, p98)
                    arr = ((arr - p2) / (p98 - p2 + 1e-8) * 255).astype(np.uint8)
                    return arr

                rgb = np.stack([normalize_band(r), normalize_band(g), normalize_band(b)], axis=-1)
                return Image.fromarray(rgb, mode="RGB")

    except ImportError:
        logger.warning("rasterio 미설치 → 일반 이미지로 파싱 시도")
        return _load_standard(data)


def _normalize_to_rgb(img: Image.Image) -> Image.Image:
    """임의 모드 이미지를 RGB로 정규화"""
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        return background
    elif img.mode == "L":
        return img.convert("RGB")
    elif img.mode in ("I", "F"):
        # 16/32비트 그레이스케일
        arr = np.array(img, dtype=np.float32)
        p2, p98 = np.percentile(arr, [2, 98])
        arr = np.clip(arr, p2, p98)
        arr = ((arr - p2) / (p98 - p2 + 1e-8) * 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    elif img.mode != "RGB":
        return img.convert("RGB")
    return img


def resize_for_model(img: Image.Image, max_size: int = None) -> Image.Image:
    """
    모델 입력 크기로 리사이즈
    종횡비 유지, 짧은 변 기준
    """
    max_size = max_size or settings.IMAGE_RESIZE
    w, h = img.size
    if max(w, h) <= max_size:
        return img
    scale = max_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)


def crop_region(img: Image.Image, bbox: dict) -> Image.Image:
    """
    픽셀 좌표 기준 영역 크롭
    bbox: {"x": int, "y": int, "width": int, "height": int}
    또는  {"x1": int, "y1": int, "x2": int, "y2": int}
    """
    if "width" in bbox:
        x1 = bbox["x"]
        y1 = bbox["y"]
        x2 = x1 + bbox["width"]
        y2 = y1 + bbox["height"]
    else:
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

    # 이미지 범위 클리핑
    w, h = img.size
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    return img.crop((x1, y1, x2, y2))


def latlon_to_pixel(img: Image.Image, lat: float, lon: float,
                    geotransform: Optional[dict] = None) -> tuple[int, int]:
    """
    위경도 → 픽셀 좌표 변환
    geotransform이 없으면 이미지 중심을 (0,0)으로 가정
    """
    if geotransform is None:
        # 메타데이터 없이 이미지 전체 범위 추정 불가
        # → 중심점 반환
        w, h = img.size
        return w // 2, h // 2

    # GDAL 스타일 geotransform: [x_origin, pixel_width, 0, y_origin, 0, pixel_height]
    x_origin = geotransform["x_origin"]
    y_origin = geotransform["y_origin"]
    pixel_width = geotransform["pixel_width"]
    pixel_height = geotransform["pixel_height"]  # 보통 음수

    px = int((lon - x_origin) / pixel_width)
    py = int((lat - y_origin) / pixel_height)
    return px, py


def validate_image_size(data: bytes) -> None:
    """파일 크기 검증"""
    max_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
    if len(data) > max_bytes:
        raise ValueError(
            f"이미지 크기 초과: {len(data) / 1024 / 1024:.1f}MB "
            f"(최대 {settings.MAX_IMAGE_SIZE_MB}MB)"
        )
