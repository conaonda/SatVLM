"""
에어갭 환경용 모델 다운로드 스크립트
인터넷 연결된 머신에서 실행 후 결과를 에어갭 환경으로 이동

사용법:
    # 1. 인터넷 연결 머신에서 (HuggingFace 다운로드)
    python scripts/download_model.py --model geochat --output ./models/

    # 2. 에어갭 머신으로 이동 후
    rsync -av ./models/ airgap-server:/opt/satellite-vlm/models/
    # 또는 외장 드라이브 사용
"""

import argparse
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


MODEL_REGISTRY = {
    "geochat": {
        "hf_path": "MBZUAI/geochat-7B",
        "local_name": "geochat-7b",
        "description": "위성영상 특화 VLM (LLaVA-1.5 기반, CVPR 2024)",
        "size_gb": 14,
        "env_key": "GEOCHAT_LOCAL_PATH",
    },
    "qwen2-vl-7b": {
        "hf_path": "Qwen/Qwen2-VL-7B-Instruct",
        "local_name": "qwen2-vl-7b",
        "description": "고성능 범용 VLM (fine-tuning 권장 기반)",
        "size_gb": 15,
        "env_key": "QWEN2_VL_LOCAL_PATH",
    },
    "internvl2-8b": {
        "hf_path": "OpenGVLab/InternVL2-8B",
        "local_name": "internvl2-8b",
        "description": "고성능 오픈소스 VLM",
        "size_gb": 17,
        "env_key": "INTERNVL2_LOCAL_PATH",
    },
    "qwen2-vl-2b": {
        "hf_path": "Qwen/Qwen2-VL-2B-Instruct",
        "local_name": "qwen2-vl-2b",
        "description": "경량 버전 (GPU 메모리 부족 시)",
        "size_gb": 5,
        "env_key": "QWEN2_VL_LOCAL_PATH",
    },
}


def download_model(model_key: str, output_base: str, hf_token: str = None) -> str:
    """HuggingFace Hub에서 모델 다운로드"""
    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"알 수 없는 모델: {model_key}\n"
            f"사용 가능: {', '.join(MODEL_REGISTRY.keys())}"
        )

    info = MODEL_REGISTRY[model_key]
    local_path = Path(output_base) / info["local_name"]

    if local_path.exists() and any(local_path.iterdir()):
        logger.info(f"이미 존재: {local_path} (건너뜀)")
        return str(local_path)

    logger.info(
        f"다운로드 시작: {info['hf_path']}\n"
        f"예상 크기: ~{info['size_gb']}GB\n"
        f"저장 위치: {local_path}"
    )

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=info["hf_path"],
            local_dir=str(local_path),
            token=hf_token,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],  # TF/Flax 제외
        )
        logger.info(f"✅ 다운로드 완료: {local_path}")

        # .env 업데이트 안내
        print(f"\n.env 파일에 추가하세요:")
        print(f"  {info['env_key']}={local_path}")

        return str(local_path)

    except ImportError:
        logger.error("huggingface_hub 미설치\n  pip install huggingface_hub")
        raise


def check_gpu_compatibility(model_key: str) -> None:
    """GPU 메모리 호환성 확인"""
    try:
        import torch
        if not torch.cuda.is_available():
            logger.warning("⚠️  CUDA 미감지 - CPU 모드로 동작 (매우 느림)")
            return

        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        model_size = MODEL_REGISTRY.get(model_key, {}).get("size_gb", 0)
        required = model_size * 1.2  # 20% 여유

        if gpu_mem < required:
            logger.warning(
                f"⚠️  GPU 메모리 부족 가능성\n"
                f"   GPU: {gpu_mem:.1f}GB, 필요: ~{required:.1f}GB\n"
                f"   4bit 양자화 옵션 고려: --quantize int4"
            )
        else:
            logger.info(f"✅ GPU 호환: {gpu_mem:.1f}GB (모델 요구 ~{required:.1f}GB)")

    except ImportError:
        logger.warning("torch 미설치, GPU 확인 생략")


def main():
    parser = argparse.ArgumentParser(
        description="에어갭 환경용 VLM 모델 다운로드",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
모델 목록:
  geochat      - 위성영상 특화 (즉시 사용, CVPR 2024, ~14GB)
  qwen2-vl-7b  - 고성능 범용 (fine-tuning 권장, ~15GB)
  internvl2-8b - 고성능 오픈소스 (~17GB)
  qwen2-vl-2b  - 경량 버전 (GPU < 8GB, ~5GB)

예시:
  python scripts/download_model.py --model geochat --output ./models
  python scripts/download_model.py --model qwen2-vl-7b --output ./models --token hf_xxx
        """
    )
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--output", default="./models", help="저장 디렉토리")
    parser.add_argument("--token", help="HuggingFace token (private 모델용)")
    parser.add_argument("--list", action="store_true", help="모델 목록 출력")

    args = parser.parse_args()

    if args.list:
        print("\n사용 가능한 모델:")
        for key, info in MODEL_REGISTRY.items():
            print(f"  {key:<20} {info['size_gb']:>4}GB  {info['description']}")
        return

    check_gpu_compatibility(args.model)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    download_model(args.model, args.output, hf_token=args.token)


if __name__ == "__main__":
    main()
