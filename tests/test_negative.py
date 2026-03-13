"""
음성 테스트 (Negative Tests)
- 손상된 파일, 빈 파일, 지원하지 않는 포맷
- 필수 파라미터 누락
- 극단적 입력값
"""

import io
import pytest
from PIL import Image


# ============================================================
# 1. 손상/비정상 파일 업로드
# ============================================================


def test_corrupted_file(client):
    """손상된 바이너리 데이터 업로드 → 422"""
    corrupted = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # 불완전한 PNG 헤더
    resp = client.post(
        "/api/v1/describe",
        files={"image": ("broken.png", corrupted, "image/png")},
    )
    assert resp.status_code == 422


def test_empty_file(client):
    """0바이트 파일 업로드 → 422"""
    resp = client.post(
        "/api/v1/describe",
        files={"image": ("empty.jpg", b"", "image/jpeg")},
    )
    assert resp.status_code == 422


def test_random_bytes(client):
    """무작위 바이트 업로드 → 422"""
    import random
    random_data = bytes(random.getrandbits(8) for _ in range(1024))
    resp = client.post(
        "/api/v1/describe",
        files={"image": ("random.dat", random_data, "image/jpeg")},
    )
    assert resp.status_code == 422


def test_text_file_as_image(client):
    """텍스트 파일을 이미지로 업로드 → 422"""
    text = b"This is not an image file at all."
    resp = client.post(
        "/api/v1/describe",
        files={"image": ("readme.txt", text, "text/plain")},
    )
    assert resp.status_code == 422


def test_truncated_jpeg(client):
    """잘린 JPEG 업로드 → 422"""
    img = Image.new("RGB", (100, 100), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    full_bytes = buf.getvalue()
    # JPEG의 절반만 전송
    truncated = full_bytes[: len(full_bytes) // 2]
    resp = client.post(
        "/api/v1/describe",
        files={"image": ("truncated.jpg", truncated, "image/jpeg")},
    )
    # 잘린 JPEG은 Pillow가 부분 로드할 수 있어 200 또는 422 가능
    assert resp.status_code in (200, 422)


# ============================================================
# 2. 파일 크기 초과
# ============================================================


def test_oversized_file_describe(client):
    """50MB 초과 파일 → 413"""
    from unittest.mock import patch
    with patch("app.utils.image_utils.settings") as mock_settings:
        mock_settings.MAX_IMAGE_SIZE_MB = 0.001  # 1KB 제한
        big_img = Image.new("RGB", (500, 500), color="red")
        buf = io.BytesIO()
        big_img.save(buf, format="JPEG")
        resp = client.post(
            "/api/v1/describe",
            files={"image": ("big.jpg", buf.getvalue(), "image/jpeg")},
        )
        assert resp.status_code == 413


def test_oversized_file_cloud(client):
    """cloud 엔드포인트에서도 크기 초과 거부"""
    from unittest.mock import patch
    with patch("app.utils.image_utils.settings") as mock_settings:
        mock_settings.MAX_IMAGE_SIZE_MB = 0.001
        img = Image.new("RGB", (500, 500))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        resp = client.post(
            "/api/v1/cloud",
            files={"image": ("big.jpg", buf.getvalue(), "image/jpeg")},
        )
        assert resp.status_code == 413


# ============================================================
# 3. 필수 파라미터 누락
# ============================================================


def test_describe_no_image(client):
    """이미지 없이 describe 요청 → 422"""
    resp = client.post("/api/v1/describe")
    assert resp.status_code == 422


def test_compare_missing_image2(client, fake_jpeg_bytes):
    """compare에 이미지 1장만 전송 → 422"""
    resp = client.post(
        "/api/v1/compare",
        files={"image1": ("a.jpg", fake_jpeg_bytes, "image/jpeg")},
        data={"mode": "temporal"},
    )
    assert resp.status_code == 422


def test_compare_missing_mode(client, fake_jpeg_bytes):
    """compare에 mode 누락 → temporal 기본값으로 200 (Form default)"""
    resp = client.post(
        "/api/v1/compare",
        files={
            "image1": ("a.jpg", fake_jpeg_bytes, "image/jpeg"),
            "image2": ("b.jpg", fake_jpeg_bytes, "image/jpeg"),
        },
    )
    # mode의 기본값이 "temporal"이므로 200
    assert resp.status_code == 200
    assert resp.json()["mode"] == "temporal"


def test_segment_point_missing_coordinates(client, fake_jpeg_bytes):
    """segment/point에 좌표 누락 → 422"""
    resp = client.post(
        "/api/v1/segment/point",
        files={"image": ("sat.jpg", fake_jpeg_bytes, "image/jpeg")},
        # x, y 누락
    )
    assert resp.status_code == 422


def test_segment_detect_missing_target(client, fake_jpeg_bytes):
    """segment/detect에 target 누락 → 422"""
    resp = client.post(
        "/api/v1/segment/detect",
        files={"image": ("sat.jpg", fake_jpeg_bytes, "image/jpeg")},
        # target 누락
    )
    assert resp.status_code == 422


def test_cloud_no_image(client):
    """cloud에 이미지 없이 요청 → 422"""
    resp = client.post("/api/v1/cloud")
    assert resp.status_code == 422


# ============================================================
# 4. 극단적 입력값
# ============================================================


def test_segment_point_negative_coords(client, fake_jpeg_bytes):
    """음수 좌표 → 422 (범위 밖)"""
    resp = client.post(
        "/api/v1/segment/point",
        files={"image": ("sat.jpg", fake_jpeg_bytes, "image/jpeg")},
        data={"x": -1, "y": -1, "radius": 10},
    )
    assert resp.status_code == 422


def test_segment_point_zero_radius(client, fake_jpeg_bytes):
    """radius=0 → 여전히 처리 가능해야 함"""
    resp = client.post(
        "/api/v1/segment/point",
        files={"image": ("sat.jpg", fake_jpeg_bytes, "image/jpeg")},
        data={"x": 50, "y": 40, "radius": 0},
    )
    # radius=0은 빈 영역이지만 에러는 아닐 수 있음
    assert resp.status_code in (200, 422)


def test_describe_very_long_context(client, fake_jpeg_bytes):
    """매우 긴 context 문자열 (10,000자)"""
    long_context = "A" * 10000
    resp = client.post(
        "/api/v1/describe",
        files={"image": ("test.jpg", fake_jpeg_bytes, "image/jpeg")},
        data={"context": long_context},
    )
    # 서버가 crash하지 않아야 함
    assert resp.status_code == 200


def test_describe_unicode_context(client, fake_jpeg_bytes):
    """유니코드(한국어/이모지) context"""
    resp = client.post(
        "/api/v1/describe",
        files={"image": ("test.jpg", fake_jpeg_bytes, "image/jpeg")},
        data={"context": "태풍 힌남노 이후 부산항 🌊🏢"},
    )
    assert resp.status_code == 200


def test_detect_special_chars_target(client, fake_jpeg_bytes):
    """특수문자 포함 target"""
    resp = client.post(
        "/api/v1/segment/detect",
        files={"image": ("sat.jpg", fake_jpeg_bytes, "image/jpeg")},
        data={"target": "ships <script>alert('xss')</script>"},
    )
    # XSS 시도지만 서버는 crash하지 않아야 함
    assert resp.status_code == 200
    # target은 그대로 반환되어야 함 (서버 측 이스케이프 여부와 무관)
    assert resp.json()["target_object"] == "ships <script>alert('xss')</script>"
