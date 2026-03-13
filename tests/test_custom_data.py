"""
CustomData 디렉토리의 사용자 이미지를 사용한 통합 테스트

이미지 구성:
  test_image1.png   1800x924   RGBA   3.0MB  (투명도 포함)
  test_image3.webp   350x309   RGB    0.0MB  (WebP 포맷)
  test_image5.jpg   2667x1667  RGB    2.2MB  (대형 이미지)
  test_image6.jpg    600x450   RGB    0.2MB  (소형 JPEG)
  test_image7.png    587x371   RGB    0.5MB
  test_image8.png    966x721   RGB    1.8MB
  test_image9.png    966x751   RGB    1.2MB

사용법:
  pytest tests/test_custom_data.py -v -s
"""

import os
import pytest
from PIL import Image
from app.utils.image_utils import (
    load_image_from_bytes,
    resize_for_model,
    crop_region,
    validate_image_size,
)

CUSTOM_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "CustomData")

ALL_FILES = [
    "test_image1.png",   # RGBA 1800x924
    "test_image3.webp",  # WebP 350x309
    "test_image5.jpg",   # 대형 2667x1667
    "test_image6.jpg",   # 소형 600x450
    "test_image7.png",   # 587x371
    "test_image8.png",   # 966x721
    "test_image9.png",   # 966x751
]


@pytest.fixture(scope="module")
def custom_data_available():
    if not os.path.isdir(CUSTOM_DIR):
        pytest.skip("CustomData 디렉토리가 없습니다.")
    for f in ALL_FILES:
        if not os.path.isfile(os.path.join(CUSTOM_DIR, f)):
            pytest.skip(f"{f} 파일이 없습니다.")


def _read(filename):
    with open(os.path.join(CUSTOM_DIR, filename), "rb") as f:
        return f.read()


# ============================================================
# 1. 이미지 로딩 - 모든 파일이 RGB로 변환되는지
# ============================================================


@pytest.mark.parametrize("filename", ALL_FILES)
def test_load_all_formats(custom_data_available, filename):
    """모든 이미지가 RGB PIL Image로 로드되는지"""
    data = _read(filename)
    img = load_image_from_bytes(data, filename)
    assert img.mode == "RGB", f"{filename}: mode={img.mode}, RGB 아님"
    assert img.size[0] > 0 and img.size[1] > 0
    print(f"  {filename}: {img.size[0]}x{img.size[1]} RGB")


# ============================================================
# 2. RGBA → RGB 변환 (test_image1.png)
# ============================================================


def test_rgba_to_rgb_conversion(custom_data_available):
    """RGBA 이미지(test_image1.png)가 RGB로 정상 변환되는지"""
    data = _read("test_image1.png")
    raw = Image.open(os.path.join(CUSTOM_DIR, "test_image1.png"))
    assert raw.mode == "RGBA", "원본이 RGBA가 아님"

    img = load_image_from_bytes(data, "test_image1.png")
    assert img.mode == "RGB"
    assert img.size == raw.size
    print(f"  RGBA({raw.size}) → RGB({img.size})")


# ============================================================
# 3. WebP 포맷 처리 (test_image3.webp)
# ============================================================


def test_webp_format_load(custom_data_available):
    """WebP 포맷이 정상 로드되는지"""
    data = _read("test_image3.webp")
    img = load_image_from_bytes(data, "test_image3.webp")
    assert img.mode == "RGB"
    assert img.size == (350, 309)
    print(f"  WebP: {img.size}")


# ============================================================
# 4. 대형 이미지 리사이즈 (test_image5.jpg: 2667x1667)
# ============================================================


def test_resize_large_image(custom_data_available):
    """2667x1667 이미지가 1024 이내로 축소되는지"""
    data = _read("test_image5.jpg")
    img = load_image_from_bytes(data, "test_image5.jpg")
    assert img.size == (2667, 1667)

    resized = resize_for_model(img, max_size=1024)
    assert max(resized.size) <= 1024
    # 종횡비 유지
    orig_ratio = img.size[0] / img.size[1]
    new_ratio = resized.size[0] / resized.size[1]
    assert abs(orig_ratio - new_ratio) < 0.02
    print(f"  {img.size} → {resized.size}")


def test_small_image_no_resize(custom_data_available):
    """600x450 이미지는 리사이즈 불필요"""
    data = _read("test_image6.jpg")
    img = load_image_from_bytes(data, "test_image6.jpg")
    resized = resize_for_model(img, max_size=1024)
    assert resized.size == img.size
    print(f"  {img.size} → 변경 없음")


# ============================================================
# 5. 크롭 테스트
# ============================================================


def test_crop_center_region(custom_data_available):
    """각 이미지 중심 200x200 크롭"""
    for filename in ALL_FILES:
        data = _read(filename)
        img = load_image_from_bytes(data, filename)
        w, h = img.size
        cx, cy = w // 2, h // 2
        half = min(100, w // 2, h // 2)

        cropped = crop_region(img, {
            "x1": cx - half, "y1": cy - half,
            "x2": cx + half, "y2": cy + half,
        })
        assert cropped.size[0] == half * 2
        assert cropped.size[1] == half * 2


# ============================================================
# 6. 파일 크기 검증
# ============================================================


def test_all_files_within_size_limit(custom_data_available):
    """모든 파일이 50MB 이내"""
    for filename in ALL_FILES:
        data = _read(filename)
        validate_image_size(data)


# ============================================================
# 7. API 엔드포인트 테스트 - 다양한 이미지
# ============================================================


def test_api_describe_rgba_image(client, custom_data_available):
    """RGBA 이미지(test_image1.png)로 /describe 호출"""
    data = _read("test_image1.png")
    resp = client.post(
        "/api/v1/describe",
        files={"image": ("test_image1.png", data, "image/png")},
        data={"context": "satellite imagery with transparency"},
    )
    assert resp.status_code == 200
    result = resp.json()
    assert len(result["description"]) > 20
    # RGBA→RGB 변환 후 크기 유지
    assert result["image_size"][0] > 0
    print(f"  describe(RGBA): size={result['image_size']}")


def test_api_describe_webp_image(client, custom_data_available):
    """WebP 이미지(test_image3.webp)로 /describe 호출"""
    data = _read("test_image3.webp")
    resp = client.post(
        "/api/v1/describe",
        files={"image": ("test_image3.webp", data, "image/webp")},
    )
    assert resp.status_code == 200
    result = resp.json()
    assert result["image_size"] == [350, 309]
    print(f"  describe(WebP): size={result['image_size']}")


def test_api_describe_large_image(client, custom_data_available):
    """대형 이미지(2667x1667)가 리사이즈된 후 처리되는지"""
    data = _read("test_image5.jpg")
    resp = client.post(
        "/api/v1/describe",
        files={"image": ("test_image5.jpg", data, "image/jpeg")},
    )
    assert resp.status_code == 200
    result = resp.json()
    # 1024 이내로 리사이즈 되었어야 함
    assert max(result["image_size"]) <= 1024
    print(f"  describe(2667x1667→resized): size={result['image_size']}")


def test_api_cloud_multiple_images(client, custom_data_available):
    """여러 이미지로 /cloud 호출하여 모두 정상 처리되는지"""
    for filename in ["test_image6.jpg", "test_image7.png", "test_image8.png"]:
        data = _read(filename)
        resp = client.post(
            "/api/v1/cloud",
            files={"image": (filename, data, "image/jpeg")},
        )
        assert resp.status_code == 200, f"{filename} 실패: {resp.status_code}"
        assert resp.json()["cloud_coverage_estimate"] is not None
    print(f"  cloud: 3개 이미지 모두 정상")


def test_api_compare_different_sizes(client, custom_data_available):
    """크기가 다른 이미지 2장(대형 vs 소형) 비교"""
    data_large = _read("test_image5.jpg")  # 2667x1667
    data_small = _read("test_image6.jpg")  # 600x450
    resp = client.post(
        "/api/v1/compare",
        files={
            "image1": ("large.jpg", data_large, "image/jpeg"),
            "image2": ("small.jpg", data_small, "image/jpeg"),
        },
        data={"mode": "general"},
    )
    assert resp.status_code == 200
    sizes = resp.json()["image_sizes"]
    # 대형 이미지는 리사이즈, 소형은 유지
    assert max(sizes[0]) <= 1024
    assert sizes[1] == [600, 450]
    print(f"  compare: sizes={sizes}")


def test_api_compare_same_format(client, custom_data_available):
    """같은 포맷 PNG 2장 시계열 비교"""
    data1 = _read("test_image8.png")
    data2 = _read("test_image9.png")
    resp = client.post(
        "/api/v1/compare",
        files={
            "image1": ("img8.png", data1, "image/png"),
            "image2": ("img9.png", data2, "image/png"),
        },
        data={"mode": "temporal"},
    )
    assert resp.status_code == 200
    assert resp.json()["mode"] == "temporal"
    assert len(resp.json()["image_sizes"]) == 2
    print(f"  compare(temporal PNG): sizes={resp.json()['image_sizes']}")


def test_api_segment_point_on_each_image(client, custom_data_available):
    """각 이미지 중심점에서 /segment/point 호출"""
    test_files = ["test_image6.jpg", "test_image7.png", "test_image9.png"]
    for filename in test_files:
        data = _read(filename)
        img = load_image_from_bytes(data, filename)
        cx, cy = img.size[0] // 2, img.size[1] // 2

        resp = client.post(
            "/api/v1/segment/point",
            files={"image": (filename, data, "image/jpeg")},
            data={"x": cx, "y": cy, "radius": 50},
        )
        assert resp.status_code == 200, f"{filename} 실패"
        assert resp.json()["point"]["x"] == cx
    print(f"  segment/point: {len(test_files)}개 이미지 정상")


def test_api_detect_on_various_images(client, custom_data_available):
    """다양한 이미지에서 객체 감지"""
    targets = [
        ("test_image5.jpg", "buildings"),
        ("test_image7.png", "vehicles"),
        ("test_image8.png", "roads"),
    ]
    for filename, target in targets:
        data = _read(filename)
        resp = client.post(
            "/api/v1/segment/detect",
            files={"image": (filename, data, "image/jpeg")},
            data={"target": target},
        )
        assert resp.status_code == 200
        assert resp.json()["target_object"] == target
    print(f"  detect: {len(targets)}개 이미지 x 대상 정상")


def test_api_segment_full_all_images(client, custom_data_available):
    """모든 이미지에 대해 /segment 전체 세그멘테이션"""
    for filename in ALL_FILES:
        data = _read(filename)
        resp = client.post(
            "/api/v1/segment",
            files={"image": (filename, data, "image/jpeg")},
        )
        assert resp.status_code == 200, f"{filename} 실패: {resp.status_code}"
        assert len(resp.json()["segmentation"]) > 20
    print(f"  segment: {len(ALL_FILES)}개 이미지 모두 정상")


def test_api_change_detection_png_pair(client, custom_data_available):
    """비슷한 크기의 PNG 2장으로 변화 감지"""
    data1 = _read("test_image8.png")  # 966x721
    data2 = _read("test_image9.png")  # 966x751
    img1 = load_image_from_bytes(data1, "test_image8.png")
    cx, cy = img1.size[0] // 2, img1.size[1] // 2

    resp = client.post(
        "/api/v1/segment/change",
        files={
            "image1": ("img8.png", data1, "image/png"),
            "image2": ("img9.png", data2, "image/png"),
        },
        data={"x": cx, "y": cy, "radius": 100},
    )
    assert resp.status_code == 200
    result = resp.json()
    assert result["point"]["x"] == cx
    assert len(result["description"]) > 20
    print(f"  change detection: point=({cx},{cy}), OK")
