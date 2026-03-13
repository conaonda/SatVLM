"""
실제 백엔드(Qwen2-VL-2B) + CustomData 이미지 통합 테스트
GPU 없이 CPU 모드로 실행 (느림, 이미지당 수 분 소요)

사용법:
    MODEL_BACKEND=qwen2_vl QWEN2_VL_LOCAL_PATH=./models/qwen2-vl-2b DEVICE=cpu \
        python tests/test_real_backend.py
"""

import os
import sys
import time

# 환경변수 설정 (import 전에)
os.environ.setdefault("MODEL_BACKEND", "qwen2_vl")
os.environ.setdefault("QWEN2_VL_LOCAL_PATH", "./models/qwen2-vl-2b")
os.environ.setdefault("DEVICE", "cpu")
os.environ.setdefault("TORCH_DTYPE", "float32")  # CPU는 float32 권장
os.environ.setdefault("MAX_NEW_TOKENS", "256")  # CPU에서 속도를 위해 줄임

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CUSTOM_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "CustomData")

TEST_IMAGES = [
    ("test_image6.jpg", "image/jpeg"),   # 600x450, 가장 작은 이미지
    ("test_image3.webp", "image/webp"),  # 350x309, WebP
    ("test_image7.png", "image/png"),    # 587x371
]


def read_image(filename):
    with open(os.path.join(CUSTOM_DIR, filename), "rb") as f:
        return f.read()


def main():
    from config.settings import settings
    print(f"=== 실제 백엔드 테스트 ===")
    print(f"  Backend : {settings.MODEL_BACKEND}")
    print(f"  Model   : {settings.MODEL_NAME}")
    print(f"  Path    : {settings.QWEN2_VL_LOCAL_PATH}")
    print(f"  Device  : {settings.DEVICE}")
    print(f"  Dtype   : {settings.TORCH_DTYPE}")
    print(f"  MaxTok  : {settings.MAX_NEW_TOKENS}")
    print()

    from fastapi.testclient import TestClient
    from app.main import app

    results = []

    with TestClient(app) as client:
        # 1. Health check
        print("[1/5] Health check...")
        t0 = time.time()
        resp = client.get("/health")
        elapsed = time.time() - t0
        data = resp.json()
        ok = resp.status_code == 200 and data.get("model_loaded")
        results.append(("health", ok, elapsed))
        print(f"  {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s) "
              f"model_loaded={data.get('model_loaded')} model={data.get('model')}")
        if not ok:
            print("  모델 로드 실패. 중단합니다.")
            return 1

        # 2. Describe (가장 작은 이미지)
        fname = "test_image6.jpg"
        print(f"\n[2/5] /describe ({fname})...")
        img_data = read_image(fname)
        t0 = time.time()
        resp = client.post(
            "/api/v1/describe",
            files={"image": (fname, img_data, "image/jpeg")},
            data={"context": "urban satellite imagery", "language": "en"},
        )
        elapsed = time.time() - t0
        ok = resp.status_code == 200
        results.append(("describe", ok, elapsed))
        if ok:
            desc = resp.json()["description"]
            print(f"  PASS ({elapsed:.1f}s) size={resp.json()['image_size']}")
            print(f"  응답: {desc[:200]}...")
        else:
            print(f"  FAIL ({elapsed:.1f}s) status={resp.status_code}")
            print(f"  {resp.text[:300]}")

        # 3. Cloud
        fname = "test_image7.png"
        print(f"\n[3/5] /cloud ({fname})...")
        img_data = read_image(fname)
        t0 = time.time()
        resp = client.post(
            "/api/v1/cloud",
            files={"image": (fname, img_data, "image/png")},
        )
        elapsed = time.time() - t0
        ok = resp.status_code == 200
        results.append(("cloud", ok, elapsed))
        if ok:
            rj = resp.json()
            print(f"  PASS ({elapsed:.1f}s) coverage={rj.get('cloud_coverage_estimate')} quality={rj.get('quality_score')}")
            print(f"  응답: {rj['analysis'][:200]}...")
        else:
            print(f"  FAIL ({elapsed:.1f}s) status={resp.status_code}")
            print(f"  {resp.text[:300]}")

        # 4. Segment/detect
        fname = "test_image6.jpg"
        print(f"\n[4/5] /segment/detect ({fname}, target=buildings)...")
        img_data = read_image(fname)
        t0 = time.time()
        resp = client.post(
            "/api/v1/segment/detect",
            files={"image": (fname, img_data, "image/jpeg")},
            data={"target": "buildings"},
        )
        elapsed = time.time() - t0
        ok = resp.status_code == 200
        results.append(("detect", ok, elapsed))
        if ok:
            print(f"  PASS ({elapsed:.1f}s)")
            print(f"  응답: {resp.json()['detections'][:200]}...")
        else:
            print(f"  FAIL ({elapsed:.1f}s) status={resp.status_code}")
            print(f"  {resp.text[:300]}")

        # 5. Compare (2장)
        print(f"\n[5/5] /compare (test_image6.jpg vs test_image7.png)...")
        data1 = read_image("test_image6.jpg")
        data2 = read_image("test_image7.png")
        t0 = time.time()
        resp = client.post(
            "/api/v1/compare",
            files={
                "image1": ("img1.jpg", data1, "image/jpeg"),
                "image2": ("img2.png", data2, "image/png"),
            },
            data={"mode": "general"},
        )
        elapsed = time.time() - t0
        ok = resp.status_code == 200
        results.append(("compare", ok, elapsed))
        if ok:
            print(f"  PASS ({elapsed:.1f}s) sizes={resp.json()['image_sizes']}")
            print(f"  응답: {resp.json()['comparison'][:200]}...")
        else:
            print(f"  FAIL ({elapsed:.1f}s) status={resp.status_code}")
            print(f"  {resp.text[:300]}")

    # Summary
    print("\n" + "=" * 50)
    print("결과 요약:")
    total_time = sum(r[2] for r in results)
    passed = sum(1 for r in results if r[1])
    for name, ok, t in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<12} {t:.1f}s")
    print(f"\n{passed}/{len(results)} 통과, 총 {total_time:.1f}s")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
