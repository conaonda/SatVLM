# 🛰️ Satellite VLM API

**v0.1.1** | [CHANGELOG](CHANGELOG.md) | [LICENSE](LICENSE)

위성영상 분석을 위한 로컬 REST API 서버
**완전 에어갭(air-gap) 환경 지원** — 모델을 로컬에서 서빙, 인터넷 불필요

---

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────┐
│                    에어갭 로컬 환경                        │
│                                                          │
│  클라이언트 → FastAPI (포트 8000)                         │
│               │                                          │
│               ├── /describe   단일 영상 설명              │
│               ├── /compare    영상 비교 (시계열/일반)      │
│               ├── /cloud      구름 커버리지 분석           │
│               └── /segment/*  세그멘테이션/인지            │
│                    │                                     │
│               VLM 모델 (로컬)                             │
│               GeoChat / Qwen2-VL / InternVL2             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                 Vertex AI (인터넷 연결 환경)               │
│                 Fine-tuning 전용 사용                     │
│                                                          │
│  위성영상 데이터 → Vertex AI Training Job                  │
│  LoRA Fine-tuning → GCS 저장                             │
│  gsutil 다운로드 → 에어갭 환경으로 물리 이동                │
└─────────────────────────────────────────────────────────┘
```

---

## 지원 모델

| 모델 | 특징 | 크기 | 권장 용도 |
|------|------|------|----------|
| **GeoChat** (기본값) | 위성영상 특화 VLM (CVPR 2024) | ~14GB | 즉시 사용 가능한 RS 분석 |
| **Qwen2-VL-7B** | 고성능 범용 VLM | ~15GB | Fine-tuning 후 최고 성능 |
| **InternVL2-8B** | 고성능 오픈소스 | ~17GB | 범용 대안 |
| **Qwen2-VL-2B** | 경량 버전 | ~5GB | GPU 메모리 부족 시 |

---

## 설치 및 실행

### 1단계: 모델 다운로드 (인터넷 연결 환경)

```bash
# 의존성 설치
pip install -r requirements.txt

# 위성영상 특화 GeoChat 다운로드 (즉시 사용 가능, 권장)
python scripts/download_model.py --model geochat --output ./models

# 또는 고성능 Qwen2-VL (fine-tuning 후 권장)
python scripts/download_model.py --model qwen2-vl-7b --output ./models

# 에어갭 환경으로 이동
rsync -av ./models/ airgap-server:/opt/satellite-vlm/models/
# 또는 외장 드라이브 복사
```

### 2단계: 환경 설정

```bash
cp .env.example .env
# .env 편집
nano .env
```

주요 설정:
```env
MODEL_BACKEND=geochat
GEOCHAT_LOCAL_PATH=./models/geochat-7b
DEVICE=cuda
DEFAULT_LANGUAGE=ko
```

### Mock 모드 (GPU 없이 실행)

GPU나 모델 없이 서버를 구동하여 REST API를 테스트할 수 있습니다.

```bash
# .env에 mock 백엔드 설정
echo "MODEL_BACKEND=mock" > .env

# 서버 실행 (GPU/모델 다운로드 불필요)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Mock 모드에서는 프롬프트 키워드 기반으로 현실적인 분석 응답을 생성합니다.
모든 API 엔드포인트(`/describe`, `/compare`, `/cloud`, `/segment/*`)를 정상적으로 호출할 수 있습니다.

### 3단계: 서버 실행

```bash
# 개발 모드
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 프로덕션 모드
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

# 백그라운드 실행
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
```

---

## API 엔드포인트

### `GET /health` — 서버 상태 확인

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model": "GeoChat-7B (Remote Sensing)",
  "model_loaded": true,
  "device": "cuda"
}
```

---

### `POST /api/v1/describe` — 단일 영상 설명

```bash
curl -X POST http://localhost:8000/api/v1/describe \
  -F "image=@satellite.tif" \
  -F "context=태풍 힌남노 이후 포항 지역" \
  -F "language=ko"
```

**응답:**
```json
{
  "description": "이 위성영상은 포항 지역을 촬영한 것으로...",
  "model": "GeoChat-7B (Remote Sensing)",
  "image_size": [1024, 768],
  "processing_time_ms": 3420.5
}
```

---

### `POST /api/v1/compare` — 영상 비교

```bash
# 시계열 변화 감지
curl -X POST http://localhost:8000/api/v1/compare \
  -F "image1=@before_flood.tif" \
  -F "image2=@after_flood.tif" \
  -F "mode=temporal" \
  -F "context=2023년 오송 지하차도 침수 사고 전후"

# 일반 비교
curl -X POST http://localhost:8000/api/v1/compare \
  -F "image1=@scene_a.tif" \
  -F "image2=@scene_b.tif" \
  -F "mode=general"
```

---

### `POST /api/v1/cloud` — 구름 커버리지 분석

```bash
curl -X POST http://localhost:8000/api/v1/cloud \
  -F "image=@sentinel2_korea.tif"
```

**응답:**
```json
{
  "analysis": "전체 구름 커버리지: 약 35%...",
  "cloud_coverage_estimate": 35.0,
  "quality_score": 6,
  "model": "GeoChat-7B (Remote Sensing)",
  "processing_time_ms": 2890.1
}
```

---

### `POST /api/v1/segment` — 토지피복 세그멘테이션

```bash
curl -X POST http://localhost:8000/api/v1/segment \
  -F "image=@urban_area.tif"
```

---

### `POST /api/v1/segment/point` — 특정 픽셀 지점 설명

```bash
curl -X POST http://localhost:8000/api/v1/segment/point \
  -F "image=@scene.tif" \
  -F "x=512" \
  -F "y=384" \
  -F "radius=100"
```

---

### `POST /api/v1/segment/region` — 바운딩박스 영역 설명

```bash
curl -X POST http://localhost:8000/api/v1/segment/region \
  -F "image=@scene.tif" \
  -F "x1=200" -F "y1=300" -F "x2=600" -F "y2=700"
```

---

### `POST /api/v1/segment/detect` — 특정 객체 감지

```bash
# 선박 감지
curl -X POST http://localhost:8000/api/v1/segment/detect \
  -F "image=@port.tif" -F "target=선박"

# 건물 감지
curl -X POST http://localhost:8000/api/v1/segment/detect \
  -F "image=@urban.tif" -F "target=buildings"
```

---

### `POST /api/v1/segment/change` — 특정 지점 변화 감지

```bash
curl -X POST http://localhost:8000/api/v1/segment/change \
  -F "image1=@before.tif" \
  -F "image2=@after.tif" \
  -F "x=512" -F "y=384" -F "radius=150"
```

---

## Vertex AI Fine-tuning 워크플로우

```bash
# 1. 어노테이션 데이터 준비 (JSONL 형식)
# annotation.jsonl 예시:
# {"image": "img_001.jpg", "task": "describe", "answer": "도시 지역..."}
# {"image": "img_002.jpg", "task": "cloud", "answer": "구름 35%..."}

# 2. Vertex AI Fine-tuning Job 제출
python scripts/vertex_finetune.py \
  --project my-gcp-project \
  --bucket gs://my-bucket \
  --base-model Qwen/Qwen2-VL-7B-Instruct \
  --dataset-gcs gs://my-bucket/training_data/ \
  --image-dir ./satellite_images/ \
  --annotation-file ./annotation.jsonl

# 3. 완료 후 로컬 export (인터넷 연결 환경)
gsutil -m cp -r gs://my-bucket/models/satellite-vlm/* ./models/satellite-vlm-finetuned/

# 4. 에어갭 환경으로 이동 후 .env 설정
MODEL_BACKEND=qwen2_vl
QWEN2_VL_LOCAL_PATH=./models/satellite-vlm-finetuned

# 5. 서버 재시작
```

---

## 테스트

GPU/모델 없이 mock 기반으로 전체 테스트를 실행할 수 있습니다.

```bash
# 전체 테스트 (mock 기반, GPU 불필요)
pytest tests/ -v

# 특정 테스트 파일
pytest tests/test_describe.py -v

# CustomData 이미지 테스트 (tests/fixtures/CustomData/ 필요)
pytest tests/test_custom_data.py -v -s

# UC Merced 데이터셋 테스트 (tests/fixtures/UCMerced_LandUse/ 필요)
pytest tests/test_ucmerced_dataset.py -v -s
```

---

## API 문서

서버 실행 후 브라우저에서:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## GPU 요구사항

| 모델 | 최소 VRAM | 권장 VRAM |
|------|-----------|-----------|
| Qwen2-VL-2B | 6GB | 8GB |
| GeoChat-7B | 14GB | 16GB |
| Qwen2-VL-7B | 16GB | 24GB |
| InternVL2-8B | 18GB | 24GB |

> VRAM 부족 시 `TORCH_DTYPE=float16` 또는 4bit 양자화 (`bitsandbytes`) 사용
