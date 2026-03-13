"""
실제 이미지 파일을 사용한 통합 테스트
- 모델 추론은 여전히 mock (GPU 불필요)
- 이미지 로딩/리사이즈/크롭 등 전처리 파이프라인을 실제 이미지로 검증

사용법:
  pytest tests/test_with_real_image.py -v -s
  pytest tests/test_with_real_image.py -v -s --image "E:/temp/2024-09-10_14-43-43_1_optical.png"
"""

import os
from PIL import Image
from app.utils.image_utils import (
    load_image_from_bytes,
    resize_for_model,
    crop_region,
    validate_image_size,
)


# ============================================================
# 1. 이미지 유틸리티 테스트 (실제 이미지)
# ============================================================


def test_load_real_image(real_image_bytes, real_image_path):
    """실제 이미지를 로드하여 RGB PIL Image로 변환"""
    img = load_image_from_bytes(real_image_bytes, os.path.basename(real_image_path))
    assert img.mode == "RGB"
    assert img.size[0] > 0 and img.size[1] > 0
    print(f"  로드 완료: {img.size[0]}x{img.size[1]} ({img.mode})")


def test_resize_real_image(real_image_bytes, real_image_path):
    """실제 이미지 리사이즈 (1024px 이내)"""
    img = load_image_from_bytes(real_image_bytes, os.path.basename(real_image_path))
    resized = resize_for_model(img, max_size=1024)
    assert max(resized.size) <= 1024
    orig_ratio = img.size[0] / img.size[1]
    new_ratio = resized.size[0] / resized.size[1]
    assert abs(orig_ratio - new_ratio) < 0.02
    print(f"  리사이즈: {img.size} -> {resized.size}")


def test_crop_center_of_real_image(real_image_bytes, real_image_path):
    """실제 이미지 중심 영역 크롭"""
    img = load_image_from_bytes(real_image_bytes, os.path.basename(real_image_path))
    w, h = img.size
    cx, cy = w // 2, h // 2
    size = min(w, h) // 4

    cropped = crop_region(img, {
        "x1": cx - size, "y1": cy - size,
        "x2": cx + size, "y2": cy + size,
    })
    assert cropped.size[0] == size * 2
    assert cropped.size[1] == size * 2
    print(f"  크롭: 중심({cx},{cy})에서 {size*2}x{size*2}")


def test_validate_real_image_size(real_image_bytes):
    """실제 이미지 크기 검증 (50MB 이내)"""
    validate_image_size(real_image_bytes)
    mb = len(real_image_bytes) / 1024 / 1024
    print(f"  파일 크기: {mb:.1f}MB (제한: 50MB)")


# ============================================================
# 2. API 엔드포인트 테스트 (실제 이미지 + mock 추론)
# ============================================================


def test_api_describe_real_image(client, real_image_bytes, real_image_path):
    """실제 이미지로 /describe API 호출"""
    filename = os.path.basename(real_image_path)
    resp = client.post(
        "/api/v1/describe",
        files={"image": (filename, real_image_bytes, "image/png")},
        data={"context": "광학 위성영상"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "description" in data
    assert data["image_size"][0] > 0
    print(f"  describe: image_size={data['image_size']}, time={data['processing_time_ms']}ms")


def test_api_cloud_real_image(client, real_image_bytes, real_image_path):
    """실제 이미지로 /cloud API 호출"""
    filename = os.path.basename(real_image_path)
    resp = client.post(
        "/api/v1/cloud",
        files={"image": (filename, real_image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "analysis" in data
    print(f"  cloud: coverage={data['cloud_coverage_estimate']}%, quality={data['quality_score']}")


def test_api_segment_point_real_image(client, real_image_bytes, real_image_path):
    """실제 이미지로 /segment/point API 호출 (이미지 중심점)"""
    img = Image.open(real_image_path)
    cx, cy = img.size[0] // 2, img.size[1] // 2

    filename = os.path.basename(real_image_path)
    resp = client.post(
        "/api/v1/segment/point",
        files={"image": (filename, real_image_bytes, "image/png")},
        data={"x": cx, "y": cy, "radius": 100},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["point"]["x"] == cx
    assert data["point"]["y"] == cy
    print(f"  segment/point: point=({cx},{cy}), time={data['processing_time_ms']}ms")


def test_api_detect_real_image(client, real_image_bytes, real_image_path):
    """실제 이미지로 /segment/detect API 호출"""
    filename = os.path.basename(real_image_path)
    resp = client.post(
        "/api/v1/segment/detect",
        files={"image": (filename, real_image_bytes, "image/png")},
        data={"target": "buildings"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["target_object"] == "buildings"
    print(f"  detect: target={data['target_object']}, time={data['processing_time_ms']}ms")


def test_api_compare_real_image(client, real_image_bytes, real_image_path):
    """실제 이미지 2장(같은 이미지)으로 /compare API 호출"""
    filename = os.path.basename(real_image_path)
    resp = client.post(
        "/api/v1/compare",
        files={
            "image1": (filename, real_image_bytes, "image/png"),
            "image2": (filename, real_image_bytes, "image/png"),
        },
        data={"mode": "temporal"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "temporal"
    assert len(data["image_sizes"]) == 2
    print(f"  compare: sizes={data['image_sizes']}, time={data['processing_time_ms']}ms")
