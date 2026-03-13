"""POST /api/v1/describe 테스트 (응답 검증 강화)"""


def test_describe_success(client, fake_jpeg_bytes):
    """정상 요청: 응답 구조 및 내용 검증"""
    resp = client.post(
        "/api/v1/describe",
        files={"image": ("test.jpg", fake_jpeg_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()

    # 구조 검증
    assert "description" in data
    assert "model" in data
    assert "image_size" in data
    assert "processing_time_ms" in data

    # 내용 검증 - 빈 응답이 아닌지, 의미 있는 길이인지
    assert len(data["description"]) > 20, "description이 너무 짧음"
    assert data["model"] != "", "model이 비어있음"
    assert data["image_size"][0] > 0 and data["image_size"][1] > 0
    assert data["processing_time_ms"] >= 0


def test_describe_with_context(client, fake_jpeg_bytes):
    """context 포함 요청"""
    resp = client.post(
        "/api/v1/describe",
        files={"image": ("test.jpg", fake_jpeg_bytes, "image/jpeg")},
        data={"context": "flood area"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["description"]) > 20


def test_describe_response_is_not_static(client, fake_jpeg_bytes):
    """describe 응답이 cloud 응답과 다른지 확인 (mock 분기 검증)"""
    resp_describe = client.post(
        "/api/v1/describe",
        files={"image": ("test.jpg", fake_jpeg_bytes, "image/jpeg")},
    )
    resp_cloud = client.post(
        "/api/v1/cloud",
        files={"image": ("test.jpg", fake_jpeg_bytes, "image/jpeg")},
    )
    assert resp_describe.json()["description"] != resp_cloud.json()["analysis"], \
        "describe와 cloud의 응답이 동일함 - mock 분기 실패"
