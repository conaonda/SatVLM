"""
UC Merced Land Use 데이터셋을 활용한 통합 테스트
- 21개 클래스의 실제 위성영상 (256x256, TIFF)
- 모델 추론은 mock (GPU 불필요)
- 이미지 전처리 파이프라인 + API 엔드포인트를 실제 위성영상으로 검증

사용법:
  pytest tests/test_ucmerced_dataset.py -v -s
"""

import os
import pytest
from PIL import Image
from app.utils.image_utils import (
    load_image_from_bytes,
    resize_for_model,
    crop_region,
    validate_image_size,
    _normalize_to_rgb,
)

DATASET_ROOT = os.path.join(
    os.path.dirname(__file__), "fixtures", "UCMerced_LandUse", "Images"
)

# 테스트에 사용할 대표 클래스 및 이미지
SAMPLE_CLASSES = [
    "agricultural",
    "airplane",
    "buildings",
    "beach",
    "forest",
    "harbor",
    "river",
    "runway",
    "storagetanks",
    "denseresidential",
]


def _get_image_path(cls_name, idx=0):
    return os.path.join(DATASET_ROOT, cls_name, f"{cls_name}{idx:02d}.tif")


def _read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


@pytest.fixture(scope="module")
def dataset_available():
    if not os.path.isdir(DATASET_ROOT):
        pytest.skip("UC Merced 데이터셋이 없습니다. tests/fixtures/에 다운로드하세요.")


# ============================================================
# 1. 이미지 로딩 테스트 - 10개 클래스
# ============================================================


@pytest.mark.parametrize("cls_name", SAMPLE_CLASSES)
def test_load_satellite_image(dataset_available, cls_name):
    """각 클래스의 위성영상을 로드하여 RGB 변환 확인"""
    path = _get_image_path(cls_name)
    data = _read_bytes(path)
    img = load_image_from_bytes(data, os.path.basename(path))

    assert img.mode == "RGB"
    assert img.size[0] > 0 and img.size[1] > 0
    print(f"  [{cls_name}] {img.size[0]}x{img.size[1]} RGB, {len(data)/1024:.0f}KB")


# ============================================================
# 2. 전처리 파이프라인 테스트
# ============================================================


def test_resize_preserves_aspect(dataset_available):
    """리사이즈 시 종횡비 유지 확인 (256x256 → 변경 없어야 함)"""
    path = _get_image_path("buildings")
    img = load_image_from_bytes(_read_bytes(path), "buildings00.tif")
    resized = resize_for_model(img, max_size=1024)
    # 256x256은 1024 이하이므로 변경 없어야 함
    assert resized.size == img.size
    print(f"  buildings: {img.size} → {resized.size} (변경 없음)")


def test_resize_with_small_limit(dataset_available):
    """max_size=128로 제한하면 축소되어야 함"""
    path = _get_image_path("harbor")
    img = load_image_from_bytes(_read_bytes(path), "harbor00.tif")
    resized = resize_for_model(img, max_size=128)
    assert max(resized.size) <= 128
    print(f"  harbor: {img.size} → {resized.size}")


def test_crop_quadrants(dataset_available):
    """이미지를 4분면으로 크롭"""
    path = _get_image_path("airplane")
    img = load_image_from_bytes(_read_bytes(path), "airplane00.tif")
    w, h = img.size
    hw, hh = w // 2, h // 2

    quadrants = [
        {"x1": 0, "y1": 0, "x2": hw, "y2": hh},       # 좌상
        {"x1": hw, "y1": 0, "x2": w, "y2": hh},        # 우상
        {"x1": 0, "y1": hh, "x2": hw, "y2": h},        # 좌하
        {"x1": hw, "y1": hh, "x2": w, "y2": h},        # 우하
    ]
    for i, bbox in enumerate(quadrants):
        cropped = crop_region(img, bbox)
        assert cropped.size == (hw, hh)
    print(f"  airplane: {w}x{h} → 4분면 각 {hw}x{hh}")


def test_validate_all_samples_within_limit(dataset_available):
    """모든 샘플 이미지가 50MB 이하인지 확인"""
    for cls_name in SAMPLE_CLASSES:
        path = _get_image_path(cls_name)
        data = _read_bytes(path)
        validate_image_size(data)  # 예외 없어야 함


# ============================================================
# 3. API 엔드포인트 테스트 - 다양한 클래스 이미지
# ============================================================


def test_api_describe_buildings(client, dataset_available):
    """건물 위성영상으로 /describe API 테스트"""
    path = _get_image_path("buildings")
    data = _read_bytes(path)
    resp = client.post(
        "/api/v1/describe",
        files={"image": ("buildings00.tif", data, "image/tiff")},
        data={"context": "urban area with buildings"},
    )
    assert resp.status_code == 200
    result = resp.json()
    assert "description" in result
    assert result["image_size"][0] > 0
    print(f"  describe(buildings): size={result['image_size']}, time={result['processing_time_ms']}ms")


def test_api_describe_forest(client, dataset_available):
    """산림 위성영상으로 /describe API 테스트"""
    path = _get_image_path("forest")
    data = _read_bytes(path)
    resp = client.post(
        "/api/v1/describe",
        files={"image": ("forest00.tif", data, "image/tiff")},
    )
    assert resp.status_code == 200
    print(f"  describe(forest): size={resp.json()['image_size']}")


def test_api_cloud_beach(client, dataset_available):
    """해변 위성영상으로 /cloud API 테스트"""
    path = _get_image_path("beach")
    data = _read_bytes(path)
    resp = client.post(
        "/api/v1/cloud",
        files={"image": ("beach00.tif", data, "image/tiff")},
    )
    assert resp.status_code == 200
    result = resp.json()
    assert "analysis" in result
    print(f"  cloud(beach): coverage={result['cloud_coverage_estimate']}%")


def test_api_compare_agricultural(client, dataset_available):
    """농경지 영상 2장 비교 테스트"""
    data1 = _read_bytes(_get_image_path("agricultural", 0))
    data2 = _read_bytes(_get_image_path("agricultural", 1))
    resp = client.post(
        "/api/v1/compare",
        files={
            "image1": ("agricultural00.tif", data1, "image/tiff"),
            "image2": ("agricultural01.tif", data2, "image/tiff"),
        },
        data={"mode": "temporal"},
    )
    assert resp.status_code == 200
    result = resp.json()
    assert result["mode"] == "temporal"
    assert len(result["image_sizes"]) == 2
    print(f"  compare(agricultural): sizes={result['image_sizes']}")


def test_api_compare_different_classes(client, dataset_available):
    """서로 다른 클래스(도시 vs 산림) 비교 테스트"""
    data1 = _read_bytes(_get_image_path("denseresidential"))
    data2 = _read_bytes(_get_image_path("forest"))
    resp = client.post(
        "/api/v1/compare",
        files={
            "image1": ("denseresidential00.tif", data1, "image/tiff"),
            "image2": ("forest00.tif", data2, "image/tiff"),
        },
        data={"mode": "general"},
    )
    assert resp.status_code == 200
    assert resp.json()["mode"] == "general"
    print(f"  compare(urban vs forest): OK")


def test_api_segment_full_harbor(client, dataset_available):
    """항만 위성영상 전체 세그멘테이션"""
    data = _read_bytes(_get_image_path("harbor"))
    resp = client.post(
        "/api/v1/segment",
        files={"image": ("harbor00.tif", data, "image/tiff")},
    )
    assert resp.status_code == 200
    assert "segmentation" in resp.json()
    print(f"  segment(harbor): OK")


def test_api_segment_point_center(client, dataset_available):
    """활주로 영상 중심점 분석"""
    path = _get_image_path("runway")
    data = _read_bytes(path)
    img = load_image_from_bytes(data, "runway00.tif")
    cx, cy = img.size[0] // 2, img.size[1] // 2

    resp = client.post(
        "/api/v1/segment/point",
        files={"image": ("runway00.tif", data, "image/tiff")},
        data={"x": cx, "y": cy, "radius": 50},
    )
    assert resp.status_code == 200
    result = resp.json()
    assert result["point"]["x"] == cx
    assert result["point"]["y"] == cy
    print(f"  segment/point(runway): ({cx},{cy}), time={result['processing_time_ms']}ms")


def test_api_detect_storagetanks(client, dataset_available):
    """저장탱크 위성영상에서 객체 감지"""
    data = _read_bytes(_get_image_path("storagetanks"))
    resp = client.post(
        "/api/v1/segment/detect",
        files={"image": ("storagetanks00.tif", data, "image/tiff")},
        data={"target": "storage tanks"},
    )
    assert resp.status_code == 200
    result = resp.json()
    assert result["target_object"] == "storage tanks"
    print(f"  detect(storagetanks): OK")


def test_api_detect_airplane(client, dataset_available):
    """비행기 위성영상에서 항공기 감지"""
    data = _read_bytes(_get_image_path("airplane"))
    resp = client.post(
        "/api/v1/segment/detect",
        files={"image": ("airplane00.tif", data, "image/tiff")},
        data={"target": "aircraft"},
    )
    assert resp.status_code == 200
    assert resp.json()["target_object"] == "aircraft"
    print(f"  detect(airplane→aircraft): OK")


def test_api_change_detection_river(client, dataset_available):
    """강 영상 2장으로 변화 감지"""
    data1 = _read_bytes(_get_image_path("river", 0))
    data2 = _read_bytes(_get_image_path("river", 1))
    img = load_image_from_bytes(data1, "river00.tif")
    cx, cy = img.size[0] // 2, img.size[1] // 2

    resp = client.post(
        "/api/v1/segment/change",
        files={
            "image1": ("river00.tif", data1, "image/tiff"),
            "image2": ("river01.tif", data2, "image/tiff"),
        },
        data={"x": cx, "y": cy, "radius": 80},
    )
    assert resp.status_code == 200
    result = resp.json()
    assert result["point"]["x"] == cx
    print(f"  change(river): point=({cx},{cy}), time={result['processing_time_ms']}ms")


# ============================================================
# 4. 다양한 클래스 일괄 로딩 스트레스 테스트
# ============================================================


def test_batch_load_all_classes(dataset_available):
    """21개 전체 클래스의 첫 번째 이미지를 모두 로드"""
    classes_dir = DATASET_ROOT
    all_classes = sorted(os.listdir(classes_dir))
    loaded = 0

    for cls_name in all_classes:
        cls_dir = os.path.join(classes_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        first_img = sorted(os.listdir(cls_dir))[0]
        path = os.path.join(cls_dir, first_img)
        data = _read_bytes(path)
        img = load_image_from_bytes(data, first_img)
        assert img.mode == "RGB"
        loaded += 1

    assert loaded == 21
    print(f"  전체 {loaded}개 클래스 로딩 성공")
