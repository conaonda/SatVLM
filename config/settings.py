"""
환경 설정
- .env 파일 또는 환경변수로 오버라이드 가능
"""

from enum import Enum
from pydantic_settings import BaseSettings


class ModelBackend(str, Enum):
    GEOCHAT = "geochat"          # RS 전용 VLM (LLaVA-1.5 기반, CVPR 2024)
    QWEN2_VL = "qwen2_vl"        # 범용 고성능 VLM (fine-tuning 후보)
    INTERNVL2 = "internvl2"      # 고성능 오픈소스 VLM
    LLAVA = "llava"              # 범용 LLaVA (fallback)


class Settings(BaseSettings):
    # ── 모델 설정 ──────────────────────────────────────────
    # 위성영상 특화: GeoChat (기본값, 즉시 사용 가능)
    # fine-tuning 후: Qwen2-VL 7B 또는 InternVL2-8B 권장
    MODEL_BACKEND: ModelBackend = ModelBackend.GEOCHAT

    # GeoChat
    GEOCHAT_MODEL_PATH: str = "MBZUAI/geochat-7B"
    GEOCHAT_LOCAL_PATH: str = "./models/geochat-7b"

    # Qwen2-VL (fine-tuning 후 로컬 경로)
    QWEN2_VL_MODEL_PATH: str = "Qwen/Qwen2-VL-7B-Instruct"
    QWEN2_VL_LOCAL_PATH: str = "./models/qwen2-vl-7b"

    # InternVL2
    INTERNVL2_MODEL_PATH: str = "OpenGVLab/InternVL2-8B"
    INTERNVL2_LOCAL_PATH: str = "./models/internvl2-8b"

    # ── 추론 설정 ──────────────────────────────────────────
    DEVICE: str = "cuda"            # "cuda" | "cpu" | "cuda:0"
    TORCH_DTYPE: str = "float16"    # "float16" | "bfloat16" | "float32"
    MAX_NEW_TOKENS: int = 1024
    TEMPERATURE: float = 0.1        # 낮을수록 결정적 (분석 태스크에 적합)
    TOP_P: float = 0.9

    # ── 이미지 설정 ──────────────────────────────────────────
    MAX_IMAGE_SIZE_MB: int = 50
    # 모델 입력 해상도 (GeoChat: 336, Qwen2-VL: 동적)
    IMAGE_RESIZE: int = 1024        # 전처리 최대 크기
    SUPPORT_GEOTIFF: bool = True

    # ── 서버 설정 ──────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1                # GPU 1장 → 워커 1개

    # ── 응답 언어 ──────────────────────────────────────────
    DEFAULT_LANGUAGE: str = "ko"    # "ko" | "en"

    # ── Vertex AI (fine-tuning 전용) ──────────────────────
    VERTEX_PROJECT_ID: str = ""
    VERTEX_LOCATION: str = "us-central1"
    VERTEX_STAGING_BUCKET: str = ""

    @property
    def MODEL_NAME(self) -> str:
        mapping = {
            ModelBackend.GEOCHAT: "GeoChat-7B (Remote Sensing)",
            ModelBackend.QWEN2_VL: "Qwen2-VL-7B-Instruct",
            ModelBackend.INTERNVL2: "InternVL2-8B",
            ModelBackend.LLAVA: "LLaVA-1.5-7B",
        }
        return mapping[self.MODEL_BACKEND]

    @property
    def ACTIVE_MODEL_PATH(self) -> str:
        """현재 활성 모델의 로컬 경로 (없으면 HF hub path)"""
        import os
        paths = {
            ModelBackend.GEOCHAT: (self.GEOCHAT_LOCAL_PATH, self.GEOCHAT_MODEL_PATH),
            ModelBackend.QWEN2_VL: (self.QWEN2_VL_LOCAL_PATH, self.QWEN2_VL_MODEL_PATH),
            ModelBackend.INTERNVL2: (self.INTERNVL2_LOCAL_PATH, self.INTERNVL2_MODEL_PATH),
        }
        local, hub = paths.get(self.MODEL_BACKEND, ("", ""))
        return local if os.path.isdir(local) else hub

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
