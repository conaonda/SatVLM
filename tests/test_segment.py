"""POST /api/v1/segment/* 테스트 (응답 검증 강화)"""


def test_segment_full(client, fake_jpeg_bytes):
    """전체 세그멘테이션: 응답 내용 검증"""
    resp = client.post(
        "/api/v1/segment",
        files={"image": ("sat.jpg", fake_jpeg_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["segmentation"]) > 20, "segmentation이 너무 짧음"
    assert data["processing_time_ms"] >= 0


def test_segment_point(client, fake_jpeg_bytes):
    """포인트 분석: 좌표 및 응답 구조 검증"""
    resp = client.post(
        "/api/v1/segment/point",
        files={"image": ("sat.jpg", fake_jpeg_bytes, "image/jpeg")},
        data={"x": 50, "y": 40, "radius": 20},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["description"]) > 20
    assert data["point"]["x"] == 50
    assert data["point"]["y"] == 40
    assert data["point"]["radius"] == 20
    assert "bbox" in data["point"]


def test_segment_point_out_of_bounds(client, fake_jpeg_bytes):
    """좌표 범위 초과 → 422"""
    resp = client.post(
        "/api/v1/segment/point",
        files={"image": ("sat.jpg", fake_jpeg_bytes, "image/jpeg")},
        data={"x": 9999, "y": 9999},
    )
    assert resp.status_code == 422


def test_segment_point_boundary(client, fake_jpeg_bytes):
    """이미지 경계값 좌표 (0,0) 테스트"""
    resp = client.post(
        "/api/v1/segment/point",
        files={"image": ("sat.jpg", fake_jpeg_bytes, "image/jpeg")},
        data={"x": 0, "y": 0, "radius": 10},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["point"]["x"] == 0
    assert data["point"]["y"] == 0


def test_segment_point_max_boundary(client, fake_jpeg_bytes):
    """이미지 최대 경계 (99, 79) - 100x80 이미지"""
    resp = client.post(
        "/api/v1/segment/point",
        files={"image": ("sat.jpg", fake_jpeg_bytes, "image/jpeg")},
        data={"x": 99, "y": 79, "radius": 5},
    )
    assert resp.status_code == 200


def test_segment_detect(client, fake_jpeg_bytes):
    """객체 감지: target 반영 확인"""
    resp = client.post(
        "/api/v1/segment/detect",
        files={"image": ("sat.jpg", fake_jpeg_bytes, "image/jpeg")},
        data={"target": "buildings"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["target_object"] == "buildings"
    assert len(data["detections"]) > 20


def test_segment_region_invalid_bbox(client, fake_jpeg_bytes):
    """잘못된 바운딩박스 (x1 >= x2) → 422"""
    resp = client.post(
        "/api/v1/segment/region",
        files={"image": ("sat.jpg", fake_jpeg_bytes, "image/jpeg")},
        data={"x1": 50, "y1": 50, "x2": 30, "y2": 30},
    )
    assert resp.status_code == 422
