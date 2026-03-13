"""POST /api/v1/cloud 테스트 (응답 검증 강화)"""


def test_cloud_analysis(client, fake_jpeg_bytes):
    """정상 요청: 응답 구조 및 내용 검증"""
    resp = client.post(
        "/api/v1/cloud",
        files={"image": ("sat.jpg", fake_jpeg_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert len(data["analysis"]) > 20, "analysis가 너무 짧음"
    assert data["model"] != ""
    assert data["processing_time_ms"] >= 0


def test_cloud_parses_coverage(client, fake_jpeg_bytes):
    """mock 응답(42.3%, quality 6)에서 파싱되는지 확인"""
    resp = client.post(
        "/api/v1/cloud",
        files={"image": ("sat.jpg", fake_jpeg_bytes, "image/jpeg")},
    )
    data = resp.json()
    assert data["cloud_coverage_estimate"] == 42.3
    assert data["quality_score"] == 6


def test_cloud_coverage_in_valid_range(client, fake_jpeg_bytes):
    """파싱된 커버리지가 0-100 범위, quality가 1-10 범위인지"""
    resp = client.post(
        "/api/v1/cloud",
        files={"image": ("sat.jpg", fake_jpeg_bytes, "image/jpeg")},
    )
    data = resp.json()
    if data["cloud_coverage_estimate"] is not None:
        assert 0 <= data["cloud_coverage_estimate"] <= 100
    if data["quality_score"] is not None:
        assert 1 <= data["quality_score"] <= 10
