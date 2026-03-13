"""POST /api/v1/compare 테스트 (응답 검증 강화)"""


def test_compare_temporal(client, fake_jpeg_bytes):
    """시계열 비교: 응답 구조 및 내용 검증"""
    resp = client.post(
        "/api/v1/compare",
        files={
            "image1": ("before.jpg", fake_jpeg_bytes, "image/jpeg"),
            "image2": ("after.jpg", fake_jpeg_bytes, "image/jpeg"),
        },
        data={"mode": "temporal"},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert len(data["comparison"]) > 20, "comparison이 너무 짧음"
    assert data["mode"] == "temporal"
    assert len(data["image_sizes"]) == 2
    assert all(s[0] > 0 and s[1] > 0 for s in data["image_sizes"])
    assert data["processing_time_ms"] >= 0


def test_compare_general(client, fake_jpeg_bytes):
    resp = client.post(
        "/api/v1/compare",
        files={
            "image1": ("a.jpg", fake_jpeg_bytes, "image/jpeg"),
            "image2": ("b.jpg", fake_jpeg_bytes, "image/jpeg"),
        },
        data={"mode": "general"},
    )
    assert resp.status_code == 200
    assert resp.json()["mode"] == "general"
    assert len(resp.json()["comparison"]) > 20


def test_compare_invalid_mode(client, fake_jpeg_bytes):
    """잘못된 mode 값 → 422"""
    resp = client.post(
        "/api/v1/compare",
        files={
            "image1": ("a.jpg", fake_jpeg_bytes, "image/jpeg"),
            "image2": ("b.jpg", fake_jpeg_bytes, "image/jpeg"),
        },
        data={"mode": "invalid_mode"},
    )
    assert resp.status_code == 422
