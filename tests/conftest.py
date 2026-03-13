"""
공통 테스트 fixture
- torch mock 주입 (GPU/모델 불필요)
- FastAPI TestClient with mocked ModelManager
- 엔드포인트별 차별화된 mock 응답
- 테스트용 이미지 바이트 생성
"""

import io
import os
import sys
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from PIL import Image

# torch를 mock으로 주입하여 model_manager import 시 에러 방지
mock_torch = MagicMock()
mock_torch.float16 = "float16"
mock_torch.bfloat16 = "bfloat16"
mock_torch.float32 = "float32"
mock_torch.inference_mode = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
mock_torch.cuda.empty_cache = MagicMock()
sys.modules.setdefault("torch", mock_torch)
sys.modules.setdefault("torchvision", MagicMock())
sys.modules.setdefault("torchvision.transforms", MagicMock())
sys.modules.setdefault("torchvision.transforms.functional", MagicMock())


# ============================================================
# 엔드포인트별 mock 응답 (프롬프트 키워드로 분기)
# ============================================================

MOCK_RESPONSES = {
    "describe": (
        "This satellite image shows a dense urban area with high-rise buildings "
        "in the center. The northern section contains residential neighborhoods "
        "with mid-rise apartments. A river runs east-west through the southern "
        "portion. Image quality is good with minimal cloud interference. "
        "Estimated spatial resolution is approximately 0.5m."
    ),
    "compare": (
        "Significant changes detected between the two images. "
        "New construction observed in the northeast quadrant where previously "
        "there was bare land. Vegetation density in the western area has decreased "
        "by approximately 30%. The river course shows minor changes in the southern "
        "bank. Overall urban expansion is evident."
    ),
    "cloud": (
        "Cloud Analysis Results:\n"
        "- Overall cloud coverage: 42.3%\n"
        "- Thick cloud: 15% concentrated in the northeast\n"
        "- Thin cirrus: 27.3% distributed across central area\n"
        "- Cloud-free zones: southwest and southeast corners\n"
        "- Cloud types: Cumulus and Cirrus\n"
        "- Usability: Moderate - partial analysis possible\n"
        "- Quality Score: 6/10"
    ),
    "segment": (
        "Land cover classification results:\n"
        "- Urban/Built-up: 35% (central and eastern areas)\n"
        "- Forest: 25% (northwestern hills)\n"
        "- Agricultural: 20% (southern plains)\n"
        "- Water bodies: 10% (river and reservoir)\n"
        "- Bare soil: 7% (construction sites)\n"
        "- Mixed: 3%"
    ),
    "point": (
        "The marked region at the specified coordinates shows a commercial building "
        "complex with a large parking lot to the east. The building appears to be "
        "a shopping center with approximately 200m x 150m footprint. Surrounding "
        "area includes roads and smaller residential structures."
    ),
    "detect": (
        "Detection results for target objects:\n"
        "- Object 1: Located at approximately (120, 85), size ~30x15 pixels\n"
        "- Object 2: Located at approximately (200, 150), size ~25x12 pixels\n"
        "- Object 3: Located at approximately (50, 220), size ~28x14 pixels\n"
        "Total detected: 3 instances"
    ),
    "change_point": (
        "Change analysis at the specified location:\n"
        "Before: Vacant lot with sparse vegetation\n"
        "After: New building structure under construction\n"
        "Change type: Land use conversion (vegetation → built-up)\n"
        "Change magnitude: High"
    ),
}


def _route_mock_response(prompt: str, images: list, **kwargs) -> str:
    """프롬프트 내용을 기반으로 적절한 mock 응답 반환"""
    prompt_lower = prompt.lower()
    if "change" in prompt_lower and "detect" in prompt_lower:
        return MOCK_RESPONSES["change_point"]
    if "detect" in prompt_lower or "find all" in prompt_lower:
        return MOCK_RESPONSES["detect"]
    if "cloud" in prompt_lower:
        return MOCK_RESPONSES["cloud"]
    if "segmentation" in prompt_lower or "land cover" in prompt_lower:
        return MOCK_RESPONSES["segment"]
    if "compare" in prompt_lower or "change" in prompt_lower:
        return MOCK_RESPONSES["compare"]
    if "marked" in prompt_lower or "highlighted" in prompt_lower or "pixel" in prompt_lower:
        return MOCK_RESPONSES["point"]
    return MOCK_RESPONSES["describe"]


def _make_image_bytes(fmt="JPEG", size=(100, 80), color="green"):
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ============================================================
# pytest CLI 옵션
# ============================================================


def pytest_addoption(parser):
    parser.addoption(
        "--image",
        default=None,
        help="테스트에 사용할 실제 이미지 파일 경로",
    )


# ============================================================
# 실제 이미지 fixture
# ============================================================

_CANDIDATE_IMAGES = [
    r"E:\temp\2024-09-10_14-43-43_1_optical.png",
    r"E:\temp\2024-09-10_15-43-41_1_optical.png",
    r"E:\temp\2024-09-11_00-31-05_1_optical.png",
]


@pytest.fixture
def real_image_path(request):
    cli_path = request.config.getoption("--image")
    if cli_path and os.path.isfile(cli_path):
        return cli_path
    for path in _CANDIDATE_IMAGES:
        if os.path.isfile(path):
            return path
    pytest.skip("실제 이미지 파일을 찾을 수 없습니다. --image 옵션으로 경로를 지정하세요.")


@pytest.fixture
def real_image_bytes(real_image_path):
    with open(real_image_path, "rb") as f:
        return f.read()


# ============================================================
# 더미 이미지 fixture
# ============================================================


@pytest.fixture
def fake_jpeg_bytes():
    return _make_image_bytes("JPEG")


@pytest.fixture
def fake_png_bytes():
    return _make_image_bytes("PNG")


# ============================================================
# TestClient fixture (엔드포인트별 분기 mock)
# ============================================================


@pytest.fixture
def client():
    """ModelManager를 mock하여 모델 로딩 없이 TestClient 생성 (응답 분기)"""
    with patch("app.services.model_manager.ModelManager.load", new_callable=AsyncMock):
        with patch("app.services.model_manager.ModelManager.unload", new_callable=AsyncMock):
            with patch(
                "app.services.model_manager.ModelManager.infer",
                side_effect=_route_mock_response,
            ):
                from fastapi.testclient import TestClient
                from app.main import app

                with TestClient(app) as c:
                    app.state.model_manager.is_loaded = True
                    yield c


@pytest.fixture
def client_infer_error():
    """모델 추론이 실패하는 TestClient (에러 경로 테스트용)"""
    with patch("app.services.model_manager.ModelManager.load", new_callable=AsyncMock):
        with patch("app.services.model_manager.ModelManager.unload", new_callable=AsyncMock):
            with patch(
                "app.services.model_manager.ModelManager.infer",
                side_effect=RuntimeError("CUDA out of memory"),
            ):
                from fastapi.testclient import TestClient
                from app.main import app

                with TestClient(app) as c:
                    app.state.model_manager.is_loaded = True
                    yield c
