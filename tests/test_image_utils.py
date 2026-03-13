"""image_utils 단위 테스트"""

import io
import pytest
from PIL import Image
from app.utils.image_utils import (
    load_image_from_bytes,
    _normalize_to_rgb,
    resize_for_model,
    crop_region,
    validate_image_size,
)


def _make_bytes(fmt="JPEG", size=(100, 80), mode="RGB", color="green"):
    img = Image.new(mode, size, color=color)
    buf = io.BytesIO()
    if fmt == "JPEG" and mode != "RGB":
        img = img.convert("RGB")
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_load_jpeg():
    data = _make_bytes("JPEG")
    img = load_image_from_bytes(data, "test.jpg")
    assert img.mode == "RGB"
    assert img.size == (100, 80)


def test_load_png():
    data = _make_bytes("PNG")
    img = load_image_from_bytes(data, "test.png")
    assert img.mode == "RGB"
    assert img.size == (100, 80)


def test_normalize_rgba_to_rgb():
    rgba = Image.new("RGBA", (50, 50), (255, 0, 0, 128))
    result = _normalize_to_rgb(rgba)
    assert result.mode == "RGB"
    assert result.size == (50, 50)


def test_normalize_grayscale_to_rgb():
    gray = Image.new("L", (50, 50), 128)
    result = _normalize_to_rgb(gray)
    assert result.mode == "RGB"
    assert result.size == (50, 50)


def test_resize_large():
    img = Image.new("RGB", (2000, 1500))
    result = resize_for_model(img, max_size=1024)
    assert max(result.size) == 1024
    # 종횡비 유지 확인
    assert abs(result.size[0] / result.size[1] - 2000 / 1500) < 0.02


def test_resize_small_no_change():
    img = Image.new("RGB", (500, 400))
    result = resize_for_model(img, max_size=1024)
    assert result.size == (500, 400)


def test_crop_region_xywh():
    img = Image.new("RGB", (200, 200))
    result = crop_region(img, {"x": 10, "y": 20, "width": 50, "height": 60})
    assert result.size == (50, 60)


def test_crop_region_x1y1x2y2():
    img = Image.new("RGB", (200, 200))
    result = crop_region(img, {"x1": 10, "y1": 20, "x2": 60, "y2": 80})
    assert result.size == (50, 60)


def test_validate_size_ok():
    data = b"\x00" * 1000
    validate_image_size(data)  # 예외 없음


def test_validate_size_too_large():
    from unittest.mock import patch
    with patch("app.utils.image_utils.settings") as mock_settings:
        mock_settings.MAX_IMAGE_SIZE_MB = 0.001  # ~1KB
        with pytest.raises(ValueError, match="이미지 크기 초과"):
            validate_image_size(b"\x00" * 2000)
