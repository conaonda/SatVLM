"""
응답 품질 검증 테스트
- 엔드포인트별 mock 응답이 올바르게 분기되는지
- 응답 내용이 해당 엔드포인트에 적합한지
- 빈 응답, 동일 응답 등 mock 허상 방지
"""

import io
from PIL import Image


def _make_jpeg():
    img = Image.new("RGB", (100, 80), color="green")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ============================================================
# 1. 엔드포인트별 응답 차별화 검증
# ============================================================


def test_each_endpoint_returns_different_response(client):
    """각 엔드포인트가 서로 다른 mock 응답을 반환하는지"""
    jpeg = _make_jpeg()

    resp_describe = client.post(
        "/api/v1/describe",
        files={"image": ("test.jpg", jpeg, "image/jpeg")},
    )
    resp_cloud = client.post(
        "/api/v1/cloud",
        files={"image": ("test.jpg", jpeg, "image/jpeg")},
    )
    resp_segment = client.post(
        "/api/v1/segment",
        files={"image": ("test.jpg", jpeg, "image/jpeg")},
    )
    resp_detect = client.post(
        "/api/v1/segment/detect",
        files={"image": ("test.jpg", jpeg, "image/jpeg")},
        data={"target": "ships"},
    )

    responses = {
        "describe": resp_describe.json()["description"],
        "cloud": resp_cloud.json()["analysis"],
        "segment": resp_segment.json()["segmentation"],
        "detect": resp_detect.json()["detections"],
    }

    # 모든 응답이 서로 달라야 함
    values = list(responses.values())
    assert len(set(values)) == len(values), \
        f"일부 엔드포인트의 응답이 동일함: {responses}"


# ============================================================
# 2. 엔드포인트별 응답 내용 적합성
# ============================================================


def test_describe_response_contains_image_analysis(client):
    """describe 응답이 이미지 분석 관련 내용을 포함하는지"""
    jpeg = _make_jpeg()
    resp = client.post(
        "/api/v1/describe",
        files={"image": ("test.jpg", jpeg, "image/jpeg")},
    )
    text = resp.json()["description"].lower()
    # 위성영상 분석 관련 키워드가 하나 이상 포함
    keywords = ["image", "area", "building", "urban", "resolution", "satellite", "region"]
    assert any(kw in text for kw in keywords), \
        f"describe 응답에 분석 관련 키워드 없음: {text[:100]}"


def test_cloud_response_contains_coverage_info(client):
    """cloud 응답이 구름 관련 내용을 포함하는지"""
    jpeg = _make_jpeg()
    resp = client.post(
        "/api/v1/cloud",
        files={"image": ("test.jpg", jpeg, "image/jpeg")},
    )
    text = resp.json()["analysis"].lower()
    keywords = ["cloud", "coverage", "quality", "score", "%"]
    assert any(kw in text for kw in keywords), \
        f"cloud 응답에 구름 관련 키워드 없음: {text[:100]}"


def test_segment_response_contains_landcover(client):
    """segment 응답이 토지 피복 관련 내용을 포함하는지"""
    jpeg = _make_jpeg()
    resp = client.post(
        "/api/v1/segment",
        files={"image": ("test.jpg", jpeg, "image/jpeg")},
    )
    text = resp.json()["segmentation"].lower()
    keywords = ["urban", "forest", "water", "agricultural", "land", "cover", "%"]
    assert any(kw in text for kw in keywords), \
        f"segment 응답에 토지 피복 키워드 없음: {text[:100]}"


def test_detect_response_contains_objects(client):
    """detect 응답이 객체 감지 관련 내용을 포함하는지"""
    jpeg = _make_jpeg()
    resp = client.post(
        "/api/v1/segment/detect",
        files={"image": ("test.jpg", jpeg, "image/jpeg")},
        data={"target": "ships"},
    )
    text = resp.json()["detections"].lower()
    keywords = ["detect", "object", "located", "found", "total", "instance"]
    assert any(kw in text for kw in keywords), \
        f"detect 응답에 감지 관련 키워드 없음: {text[:100]}"


# ============================================================
# 3. 빈 응답 방지
# ============================================================


def test_no_empty_descriptions(client):
    """모든 주요 엔드포인트가 빈 문자열을 반환하지 않는지"""
    jpeg = _make_jpeg()

    endpoints = [
        ("/api/v1/describe", {"files": {"image": ("t.jpg", jpeg, "image/jpeg")}}),
        ("/api/v1/cloud", {"files": {"image": ("t.jpg", jpeg, "image/jpeg")}}),
        ("/api/v1/segment", {"files": {"image": ("t.jpg", jpeg, "image/jpeg")}}),
    ]

    for url, kwargs in endpoints:
        resp = client.post(url, **kwargs)
        assert resp.status_code == 200
        data = resp.json()
        # 첫 번째 문자열 필드가 비어있지 않은지
        for key, value in data.items():
            if isinstance(value, str) and key != "model":
                assert len(value.strip()) > 0, \
                    f"{url}의 {key}가 빈 문자열"


# ============================================================
# 4. 처리 시간 합리성
# ============================================================


def test_processing_time_is_reasonable(client):
    """처리 시간이 0 이상이고 비현실적으로 크지 않은지"""
    jpeg = _make_jpeg()
    resp = client.post(
        "/api/v1/describe",
        files={"image": ("test.jpg", jpeg, "image/jpeg")},
    )
    ms = resp.json()["processing_time_ms"]
    assert ms >= 0, "처리 시간이 음수"
    assert ms < 60000, "처리 시간이 60초 초과 (mock인데 비정상)"


# ============================================================
# 5. compare 엔드포인트 이미지 크기 일관성
# ============================================================


def test_compare_image_sizes_match_input(client):
    """compare 응답의 image_sizes가 입력 이미지 크기를 반영하는지"""
    img1 = Image.new("RGB", (100, 80))
    img2 = Image.new("RGB", (200, 150))
    buf1, buf2 = io.BytesIO(), io.BytesIO()
    img1.save(buf1, format="JPEG")
    img2.save(buf2, format="JPEG")

    resp = client.post(
        "/api/v1/compare",
        files={
            "image1": ("small.jpg", buf1.getvalue(), "image/jpeg"),
            "image2": ("large.jpg", buf2.getvalue(), "image/jpeg"),
        },
        data={"mode": "general"},
    )
    assert resp.status_code == 200
    sizes = resp.json()["image_sizes"]
    assert sizes[0] == [100, 80]
    assert sizes[1] == [200, 150]
