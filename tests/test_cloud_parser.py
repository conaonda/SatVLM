"""_parse_cloud_coverage 단위 테스트"""

from app.routers.cloud import _parse_cloud_coverage


def test_parse_english_coverage():
    coverage, _ = _parse_cloud_coverage("Overall cloud coverage: 35.5%")
    assert coverage == 35.5


def test_parse_quality_score():
    _, quality = _parse_cloud_coverage("Quality Score: 7/10")
    assert quality == 7


def test_parse_korean():
    text = "전체 구름: 45% 분포. 품질 점수: 8"
    coverage, quality = _parse_cloud_coverage(text)
    assert coverage == 45.0
    assert quality == 8


def test_parse_no_match():
    coverage, quality = _parse_cloud_coverage("The sky is clear and blue.")
    assert coverage is None
    assert quality is None


def test_parse_quality_clamped():
    _, quality = _parse_cloud_coverage("Quality Score: 15")
    assert quality == 10
