# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-03-13

### Added
- Mock 백엔드 (`MODEL_BACKEND=mock`): GPU/모델 없이 서버 구동 및 REST API 테스트 가능
- `ModelBackend.MOCK` enum 및 `MODEL_NAME` 매핑 추가
- 프롬프트 키워드 기반 현실적인 mock 응답 생성 (describe, compare, cloud, segment 등)

### Changed
- `model_manager.py`: `torch` import를 지연 로딩(lazy import)으로 변경 — mock 모드에서 torch 의존성 불필요

## [0.1.0] - 2026-03-13

### Added
- FastAPI REST API 서버 (위성영상 분석)
- 4개 모델 백엔드 지원 (GeoChat, Qwen2-VL, InternVL2, LLaVA)
- API 엔드포인트: describe, compare, cloud, segment (point/region/detect/change)
- GeoTIFF 지원 (rasterio)
- RGBA/WebP/대형 이미지 자동 처리
- 구름 커버리지 분석 및 품질 점수 파싱
- 종합 테스트 스위트 (129개 테스트)
  - 단위 테스트 (image_utils, prompts, cloud_parser)
  - API 통합 테스트 (모든 엔드포인트)
  - 네거티브 테스트 (손상 파일, 파라미터 누락, 극단값)
  - 에러 경로 테스트 (모델 추론 실패 시 500 응답)
  - 응답 품질 테스트
  - UC Merced 데이터셋 테스트
  - CustomData 이미지 테스트

### Fixed
- segment.py 5개 엔드포인트에 try-except 누락 버그 수정
