"""prompts 단위 테스트"""

from app.utils.prompts import (
    get_describe_prompt,
    get_compare_prompt,
    get_cloud_prompt,
    get_point_prompt,
    get_object_detection_prompt,
)


def test_describe_no_context():
    prompt = get_describe_prompt()
    assert "remote sensing" in prompt.lower() or "위성" in prompt or len(prompt) > 20
    assert "{context}" not in prompt


def test_describe_with_context():
    prompt = get_describe_prompt(context="Seoul flood area")
    assert "Seoul flood area" in prompt


def test_compare_temporal():
    prompt = get_compare_prompt("temporal")
    assert len(prompt) > 20


def test_compare_general():
    prompt = get_compare_prompt("general")
    assert len(prompt) > 20


def test_cloud_prompt():
    prompt = get_cloud_prompt()
    assert "cloud" in prompt.lower() or "구름" in prompt


def test_point_prompt_contains_coords():
    prompt = get_point_prompt(100, 200, 800, 600)
    assert "100" in prompt and "200" in prompt


def test_object_detection_prompt():
    prompt = get_object_detection_prompt("ships")
    assert "ships" in prompt.lower()
