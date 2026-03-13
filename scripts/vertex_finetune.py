"""
Vertex AI Fine-tuning 스크립트
위성영상 도메인에 맞게 VLM을 fine-tuning하고 결과를 로컬로 export

사용법:
    python scripts/vertex_finetune.py \\
        --project my-gcp-project \\
        --dataset gs://my-bucket/satellite-dataset/ \\
        --base-model Qwen/Qwen2-VL-7B-Instruct \\
        --output-dir ./models/satellite-vlm-finetuned

워크플로우:
    1. 데이터셋 준비 (JSONL 형식)
    2. Vertex AI Custom Job으로 LoRA fine-tuning
    3. 학습된 모델 GCS → 로컬 export
    4. 로컬 서버에 배포 (MODEL_BACKEND=qwen2_vl, QWEN2_VL_LOCAL_PATH 설정)
"""

import argparse
import json
import logging
import os
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ── 데이터셋 포맷 예시 ──────────────────────────────────────────────────────
DATASET_EXAMPLE = {
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "gs://my-bucket/images/train/img_001.jpg"
                },
                {
                    "type": "text",
                    "text": "이 위성영상을 분석하여 토지피복을 설명해주세요."
                }
            ]
        },
        {
            "role": "assistant",
            "content": "이 영상은 한국 수도권 지역의 도시 지역을 보여줍니다. "
                       "북쪽에는 고밀도 아파트 단지가 분포하며..."
        }
    ]
}

# Vertex AI Training Job 설정
TRAINING_CONFIG = {
    "display_name": "satellite-vlm-finetuning",
    "machine_type": "a2-highgpu-1g",   # A100 40GB
    "accelerator_type": "NVIDIA_TESLA_A100",
    "accelerator_count": 1,
    "replica_count": 1,
    # LoRA 설정 (전체 파라미터 대비 학습 효율 극대화)
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    # 학습 하이퍼파라미터
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,  # 효과적 배치 = 8
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "max_seq_length": 2048,
    "bf16": True,
    "dataloader_num_workers": 4,
}


def prepare_dataset(
    image_dir: str,
    annotation_file: str,
    output_jsonl: str,
    gcs_bucket: str
) -> None:
    """
    로컬 위성영상 + 어노테이션 → Vertex AI 학습용 JSONL 변환

    annotation_file 형식 (JSON Lines):
    {"image": "img_001.jpg", "task": "describe", "answer": "도시 지역..."}
    {"image": "img_002.jpg", "task": "cloud", "answer": "구름 커버리지 약 30%..."}
    {"image": ["before.jpg", "after.jpg"], "task": "compare", "answer": "변화 감지..."}
    """
    TASK_PROMPTS = {
        "describe": "이 위성영상을 상세히 분석하고 설명해주세요.",
        "cloud": "이 위성영상의 구름 커버리지를 분석해주세요.",
        "compare": "두 위성영상을 비교하여 변화를 설명해주세요.",
        "segment": "이 위성영상의 토지피복을 분류하여 설명해주세요.",
        "detect_ship": "이 위성영상에서 선박을 감지하고 설명해주세요.",
        "detect_building": "이 위성영상에서 건물을 감지하고 설명해주세요.",
    }

    records = []
    with open(annotation_file) as f:
        for line in f:
            ann = json.loads(line.strip())
            task = ann.get("task", "describe")
            prompt = TASK_PROMPTS.get(task, TASK_PROMPTS["describe"])

            images = ann["image"] if isinstance(ann["image"], list) else [ann["image"]]
            content = [
                {"type": "image", "image": f"{gcs_bucket}/{img}"}
                for img in images
            ]
            content.append({"type": "text", "text": prompt})

            records.append({
                "messages": [
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": ann["answer"]}
                ]
            })

    with open(output_jsonl, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"데이터셋 준비 완료: {len(records)}개 샘플 → {output_jsonl}")


def submit_vertex_job(
    project_id: str,
    location: str,
    staging_bucket: str,
    base_model: str,
    dataset_gcs: str,
    output_gcs: str,
    config: dict = None,
) -> str:
    """
    Vertex AI Custom Training Job 제출
    실제 환경에서는 google-cloud-aiplatform SDK 사용
    """
    try:
        from google.cloud import aiplatform
        from google.cloud.aiplatform import CustomJob, HyperparameterTuningJob
    except ImportError:
        logger.error(
            "google-cloud-aiplatform 미설치\n"
            "pip install google-cloud-aiplatform 실행 후 재시도"
        )
        raise

    cfg = {**TRAINING_CONFIG, **(config or {})}

    aiplatform.init(
        project=project_id,
        location=location,
        staging_bucket=staging_bucket,
    )

    # 학습 스크립트를 GCS에 업로드하는 방식 또는
    # pre-built container 사용
    job = aiplatform.CustomJob(
        display_name=cfg["display_name"],
        worker_pool_specs=[
            {
                "machine_spec": {
                    "machine_type": cfg["machine_type"],
                    "accelerator_type": cfg["accelerator_type"],
                    "accelerator_count": cfg["accelerator_count"],
                },
                "replica_count": cfg["replica_count"],
                "container_spec": {
                    # HuggingFace TRL/PEFT 기반 학습 컨테이너
                    "image_uri": "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-training-cu121.2-3.transformers.4-44.ubuntu2204.py311",
                    "args": [
                        f"--model_name_or_path={base_model}",
                        f"--dataset_path={dataset_gcs}",
                        f"--output_dir={output_gcs}",
                        f"--lora_r={cfg['lora_rank']}",
                        f"--lora_alpha={cfg['lora_alpha']}",
                        f"--num_train_epochs={cfg['num_train_epochs']}",
                        f"--per_device_train_batch_size={cfg['per_device_train_batch_size']}",
                        f"--learning_rate={cfg['learning_rate']}",
                        "--bf16=True",
                        "--use_lora=True",
                    ],
                },
            }
        ],
    )

    job.run(sync=False)
    logger.info(f"Vertex AI Job 제출 완료: {job.display_name} ({job.resource_name})")
    return job.resource_name


def export_model_to_local(
    gcs_model_path: str,
    local_output_dir: str,
) -> None:
    """
    GCS에서 학습된 모델 로컬로 다운로드

    에어갭 환경에서는 인터넷 연결된 머신에서 다운로드 후
    물리 이동 (USB/인트라넷) 방식 사용
    """
    local_path = Path(local_output_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"모델 다운로드 시작: {gcs_model_path} → {local_output_dir}")
    try:
        subprocess.run(
            ["gsutil", "-m", "cp", "-r", f"{gcs_model_path}/*", str(local_path)],
            check=True,
        )
        logger.info(f"✅ 모델 다운로드 완료: {local_output_dir}")
        logger.info(
            "\n다음 단계:\n"
            f"1. .env 파일 설정:\n"
            f"   MODEL_BACKEND=qwen2_vl\n"
            f"   QWEN2_VL_LOCAL_PATH={local_output_dir}\n"
            f"2. 서버 재시작: uvicorn app.main:app --reload\n"
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"gsutil 실패: {e}")
        logger.info(
            "수동 다운로드:\n"
            f"  gsutil -m cp -r '{gcs_model_path}/*' '{local_output_dir}/'"
        )
        raise


def main():
    parser = argparse.ArgumentParser(description="Vertex AI 위성영상 VLM Fine-tuning")
    parser.add_argument("--project", required=True, help="GCP 프로젝트 ID")
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--bucket", required=True, help="GCS 버킷 (gs://...)")
    parser.add_argument("--base-model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--dataset-gcs", required=True, help="학습 데이터 GCS 경로")
    parser.add_argument("--output-gcs", help="모델 출력 GCS 경로")
    parser.add_argument("--local-export", default="./models/satellite-vlm-finetuned",
                        help="로컬 export 경로")
    parser.add_argument("--prepare-only", action="store_true",
                        help="데이터셋 준비만 수행")

    # 데이터셋 준비 옵션
    parser.add_argument("--image-dir", help="로컬 이미지 디렉토리")
    parser.add_argument("--annotation-file", help="어노테이션 JSONL 파일")

    args = parser.parse_args()

    output_gcs = args.output_gcs or f"{args.bucket}/models/satellite-vlm"

    # 1. 데이터셋 준비 (선택)
    if args.image_dir and args.annotation_file:
        prepare_dataset(
            image_dir=args.image_dir,
            annotation_file=args.annotation_file,
            output_jsonl="./training_data.jsonl",
            gcs_bucket=args.bucket,
        )
        if args.prepare_only:
            return

    # 2. Vertex AI Job 제출
    job_name = submit_vertex_job(
        project_id=args.project,
        location=args.location,
        staging_bucket=args.bucket,
        base_model=args.base_model,
        dataset_gcs=args.dataset_gcs,
        output_gcs=output_gcs,
    )

    logger.info(
        f"\n학습 Job 제출됨: {job_name}\n"
        f"완료 후 모델 다운로드:\n"
        f"  python scripts/vertex_finetune.py --export-only "
        f"--gcs-model {output_gcs} --local-export {args.local_export}"
    )


if __name__ == "__main__":
    main()
