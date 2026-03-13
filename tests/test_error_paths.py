"""
에러 경로 테스트
- 모델 추론 실패 시 500 응답
- try-except가 있는 엔드포인트: describe, cloud, compare → HTTPException(500)
- try-except가 없는 엔드포인트: segment, segment/point, segment/detect → 전역 핸들러
"""

import io
import pytest
from PIL import Image


def _make_jpeg():
    img = Image.new("RGB", (100, 80), color="green")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ============================================================
# 1. try-except가 있는 엔드포인트 (HTTPException 반환)
# ============================================================


def test_describe_infer_failure(client_infer_error):
    """describe 추론 실패 → 500, detail에 에러 메시지 포함"""
    jpeg = _make_jpeg()
    resp = client_infer_error.post(
        "/api/v1/describe",
        files={"image": ("test.jpg", jpeg, "image/jpeg")},
    )
    assert resp.status_code == 500
    data = resp.json()
    assert "detail" in data
    assert "CUDA out of memory" in data["detail"]


def test_cloud_infer_failure(client_infer_error):
    """cloud 추론 실패 → 500"""
    jpeg = _make_jpeg()
    resp = client_infer_error.post(
        "/api/v1/cloud",
        files={"image": ("test.jpg", jpeg, "image/jpeg")},
    )
    assert resp.status_code == 500
    data = resp.json()
    assert "detail" in data


def test_compare_infer_failure(client_infer_error):
    """compare 추론 실패 → 500"""
    jpeg = _make_jpeg()
    resp = client_infer_error.post(
        "/api/v1/compare",
        files={
            "image1": ("a.jpg", jpeg, "image/jpeg"),
            "image2": ("b.jpg", jpeg, "image/jpeg"),
        },
        data={"mode": "temporal"},
    )
    assert resp.status_code == 500


# ============================================================
# 2. try-except가 없는 엔드포인트 (전역 핸들러로 전파)
#    - segment.py의 segment, segment/point, segment/detect는
#      infer() 호출에 try-except가 없어 예외가 직접 전파됨
#    - 이것 자체가 코드 품질 이슈 (발견된 버그)
# ============================================================


def test_segment_infer_failure_unhandled(client_infer_error):
    """segment 추론 실패 → 전역 핸들러 경유 500 (try-except 미비 발견)"""
    jpeg = _make_jpeg()
    resp = client_infer_error.post(
        "/api/v1/segment",
        files={"image": ("test.jpg", jpeg, "image/jpeg")},
    )
    assert resp.status_code == 500
    data = resp.json()
    # 전역 핸들러: {"error": "내부 서버 오류", "detail": "..."}
    assert "error" in data or "detail" in data


def test_segment_point_infer_failure_unhandled(client_infer_error):
    """segment/point 추론 실패 → 전역 핸들러 경유 500"""
    jpeg = _make_jpeg()
    resp = client_infer_error.post(
        "/api/v1/segment/point",
        files={"image": ("test.jpg", jpeg, "image/jpeg")},
        data={"x": 50, "y": 40, "radius": 20},
    )
    assert resp.status_code == 500


def test_segment_detect_infer_failure_unhandled(client_infer_error):
    """segment/detect 추론 실패 → 전역 핸들러 경유 500"""
    jpeg = _make_jpeg()
    resp = client_infer_error.post(
        "/api/v1/segment/detect",
        files={"image": ("test.jpg", jpeg, "image/jpeg")},
        data={"target": "buildings"},
    )
    assert resp.status_code == 500


# ============================================================
# 3. 에러 응답 구조 검증
# ============================================================


def test_error_response_structure(client_infer_error):
    """describe 에러 응답이 detail 필드를 포함하는지"""
    jpeg = _make_jpeg()
    resp = client_infer_error.post(
        "/api/v1/describe",
        files={"image": ("test.jpg", jpeg, "image/jpeg")},
    )
    assert resp.status_code == 500
    data = resp.json()
    assert "detail" in data
    assert "CUDA out of memory" in data["detail"]
