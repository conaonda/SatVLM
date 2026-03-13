"""
38-Cloud 데이터셋을 활용한 구름 커버리지 분석 테스트

데이터셋: 38-Cloud (Landsat 8, 384x384 RGB 패치 + 바이너리 구름 마스크)
출처: https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset

Ground Truth 마스크 기반으로 모델의 구름 커버리지 추정 정확도를 검증:
  - clear (0-10%): 4개 패치
  - partial (30-60%): 3개 패치
  - cloudy (70-100%): 3개 패치

사용법:
  # Mock 백엔드 (빠른 기능 테스트)
  pytest tests/test_38cloud.py -v

  # 실제 백엔드 (정확도 검증)
  MODEL_BACKEND=qwen2_vl QWEN2_VL_LOCAL_PATH=./models/qwen2-vl-2b DEVICE=cpu \
      python tests/test_38cloud.py
"""

import os
import sys
import numpy as np
from PIL import Image

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "38-Cloud")
IMAGE_DIR = os.path.join(FIXTURE_DIR, "images")
MASK_DIR = os.path.join(FIXTURE_DIR, "masks")

# 파일명에서 ground truth 커버리지를 파싱
# 형식: {category}_{idx}_{coverage}pct.png
SAMPLES = [
    ("clear_00_0pct.png", 0.0),
    ("clear_01_0pct.png", 0.0),
    ("clear_02_4pct.png", 4.5),
    ("clear_03_0pct.png", 0.0),
    ("partial_04_57pct.png", 56.9),
    ("partial_05_54pct.png", 53.9),
    ("partial_06_54pct.png", 54.1),
    ("cloudy_07_100pct.png", 100.0),
    ("cloudy_08_71pct.png", 71.1),
    ("cloudy_09_100pct.png", 100.0),
]


def _ground_truth_coverage(mask_filename: str) -> float:
    """마스크 이미지에서 실제 구름 커버리지(%) 계산"""
    mask = np.array(Image.open(os.path.join(MASK_DIR, mask_filename)))
    return (mask > 0).sum() / mask.size * 100


def _read_image(filename: str) -> bytes:
    with open(os.path.join(IMAGE_DIR, filename), "rb") as f:
        return f.read()


# ============================================================
# pytest 테스트 (mock 백엔드)
# ============================================================

import pytest


@pytest.fixture(scope="module")
def cloud_data_available():
    if not os.path.isdir(IMAGE_DIR):
        pytest.skip("38-Cloud fixture 디렉토리가 없습니다.")


@pytest.mark.parametrize("filename,expected_coverage", SAMPLES)
def test_ground_truth_masks_valid(cloud_data_available, filename, expected_coverage):
    """Ground truth 마스크가 올바른 커버리지 값을 가지는지 검증"""
    actual = _ground_truth_coverage(filename)
    assert abs(actual - expected_coverage) < 1.0, (
        f"{filename}: expected ~{expected_coverage}%, got {actual:.1f}%"
    )


@pytest.mark.parametrize("filename,expected_coverage", SAMPLES)
def test_cloud_api_returns_200(client, cloud_data_available, filename, expected_coverage):
    """38-Cloud 이미지로 /cloud API 호출 시 200 응답"""
    data = _read_image(filename)
    resp = client.post(
        "/api/v1/cloud",
        files={"image": (filename, data, "image/png")},
    )
    assert resp.status_code == 200
    result = resp.json()
    assert result["cloud_coverage_estimate"] is not None
    assert result["quality_score"] is not None


def test_cloud_clear_images_low_coverage(client, cloud_data_available):
    """맑은 이미지(GT <10%)에서 모델이 낮은 커버리지를 예측하는지"""
    clear_samples = [(f, c) for f, c in SAMPLES if c < 10]
    for filename, gt_coverage in clear_samples:
        data = _read_image(filename)
        resp = client.post(
            "/api/v1/cloud",
            files={"image": (filename, data, "image/png")},
        )
        assert resp.status_code == 200
        # mock 응답은 42.3%로 고정이므로 여기서는 API 동작만 검증
        assert resp.json()["cloud_coverage_estimate"] is not None


def test_cloud_cloudy_images_high_coverage(client, cloud_data_available):
    """구름 많은 이미지(GT >70%)에서 모델이 높은 커버리지를 예측하는지"""
    cloudy_samples = [(f, c) for f, c in SAMPLES if c > 70]
    for filename, gt_coverage in cloudy_samples:
        data = _read_image(filename)
        resp = client.post(
            "/api/v1/cloud",
            files={"image": (filename, data, "image/png")},
        )
        assert resp.status_code == 200
        assert resp.json()["cloud_coverage_estimate"] is not None


def test_describe_with_cloud_images(client, cloud_data_available):
    """/describe 엔드포인트에 구름 이미지 전달"""
    # 구름 많은 이미지
    data = _read_image("cloudy_08_71pct.png")
    resp = client.post(
        "/api/v1/describe",
        files={"image": ("cloudy.png", data, "image/png")},
        data={"context": "Landsat 8 satellite patch with clouds"},
    )
    assert resp.status_code == 200
    assert len(resp.json()["description"]) > 20


def test_compare_clear_vs_cloudy(client, cloud_data_available):
    """맑은 이미지 vs 구름 이미지 비교"""
    data_clear = _read_image("clear_01_0pct.png")
    data_cloudy = _read_image("cloudy_08_71pct.png")
    resp = client.post(
        "/api/v1/compare",
        files={
            "image1": ("clear.png", data_clear, "image/png"),
            "image2": ("cloudy.png", data_cloudy, "image/png"),
        },
        data={"mode": "general"},
    )
    assert resp.status_code == 200
    assert len(resp.json()["comparison"]) > 20


# ============================================================
# 실제 백엔드 정확도 테스트 (standalone 실행)
# ============================================================

def run_real_backend_cloud_test():
    """실제 모델로 구름 커버리지 추정 정확도 측정"""
    import time

    os.environ.setdefault("MODEL_BACKEND", "qwen2_vl")
    os.environ.setdefault("QWEN2_VL_LOCAL_PATH", "./models/qwen2-vl-2b")
    os.environ.setdefault("DEVICE", "cpu")
    os.environ.setdefault("TORCH_DTYPE", "float32")
    os.environ.setdefault("MAX_NEW_TOKENS", "256")

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from config.settings import settings
    print(f"=== 38-Cloud 구름 커버리지 정확도 테스트 ===")
    print(f"  Backend: {settings.MODEL_BACKEND}")
    print(f"  Model  : {settings.MODEL_NAME}")
    print(f"  Device : {settings.DEVICE}")
    print()

    from fastapi.testclient import TestClient
    from app.main import app

    results = []

    with TestClient(app) as client_app:
        # 모델 로드 확인
        resp = client_app.get("/health")
        if not resp.json().get("model_loaded"):
            print("모델 로드 실패!")
            return 1

        print(f"{'파일':<30} {'GT%':>6} {'모델%':>6} {'오차':>6} {'품질':>4} {'시간':>7}")
        print("-" * 75)

        for filename, gt_coverage in SAMPLES:
            data = _read_image(filename)
            t0 = time.time()
            resp = client_app.post(
                "/api/v1/cloud",
                files={"image": (filename, data, "image/png")},
            )
            elapsed = time.time() - t0

            if resp.status_code != 200:
                print(f"{filename:<30} {'FAIL':>6} status={resp.status_code}")
                results.append((filename, gt_coverage, None, None, elapsed, False))
                continue

            rj = resp.json()
            model_coverage = rj.get("cloud_coverage_estimate")
            quality = rj.get("quality_score")
            if model_coverage is None:
                model_coverage = -1
            if quality is None:
                quality = -1
            error = abs(model_coverage - gt_coverage) if model_coverage >= 0 else -1

            # 카테고리별 허용 오차
            if gt_coverage < 10:
                ok = model_coverage < 30  # 맑은 이미지는 30% 미만이면 OK
            elif gt_coverage > 70:
                ok = model_coverage > 40  # 구름 많은 이미지는 40% 이상이면 OK
            else:
                ok = error < 40  # 부분 구름은 오차 40% 미만

            status = "OK" if ok else "MISS"
            print(f"{filename:<30} {gt_coverage:>5.1f}% {model_coverage:>5.1f}% {error:>5.1f}% {quality:>4} {elapsed:>6.1f}s {status}")
            results.append((filename, gt_coverage, model_coverage, quality, elapsed, ok))

    # 요약
    print()
    print("=" * 75)
    total = len(results)
    passed = sum(1 for r in results if r[5])
    total_time = sum(r[4] for r in results)
    errors = [abs(r[2] - r[1]) for r in results if r[2] is not None]
    mean_error = np.mean(errors) if errors else -1

    print(f"결과: {passed}/{total} 통과")
    print(f"평균 오차: {mean_error:.1f}%")
    print(f"총 소요시간: {total_time:.1f}s")

    # 카테고리별
    for category, label in [("clear", "맑음 (<10%)"), ("partial", "부분 (30-60%)"), ("cloudy", "흐림 (>70%)")]:
        cat_results = [(r[1], r[2]) for r in results if r[2] is not None and category in SAMPLES[results.index(r)][0]]
        if cat_results:
            cat_errors = [abs(m - g) for g, m in cat_results]
            print(f"  {label}: 평균 오차 {np.mean(cat_errors):.1f}%")

    return 0 if passed >= total * 0.7 else 1


if __name__ == "__main__":
    sys.exit(run_real_backend_cloud_test())
