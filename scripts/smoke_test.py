"""
Smoke test for the running inference service - hits /health and both
/predict variants against a real image. Works the same whether the service
is running locally (uvicorn) or in Docker; just point --url at the right host.

Usage:
    python scripts/smoke_test.py --image path/to/cat_or_dog.jpg
    python scripts/smoke_test.py --image pet.jpg --url http://localhost:8000

This also doubles as the M4 "smoke test" script - failing loudly (non-zero
exit code) is intentional so a CI/CD pipeline can gate on it.
"""
import argparse
import base64
import sys

import requests


def check_health(base_url: str) -> None:
    resp = requests.get(f"{base_url}/health", timeout=10)
    resp.raise_for_status()
    body = resp.json()
    print(f"[health] {body}")
    if not body.get("model_loaded"):
        print("[health] WARNING: model_loaded=False - /predict calls below will fail with 503 "
              "until a trained model.pt is present.")


def check_predict_file(base_url: str, image_path: str) -> None:
    with open(image_path, "rb") as f:
        files = {"file": (image_path, f, "image/jpeg")}
        resp = requests.post(f"{base_url}/predict", files=files, timeout=30)
    resp.raise_for_status()
    print(f"[predict - file upload] {resp.json()}")


def check_predict_base64(base_url: str, image_path: str) -> None:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    resp = requests.post(f"{base_url}/predict/base64", json={"image_base64": b64}, timeout=30)
    resp.raise_for_status()
    print(f"[predict - base64] {resp.json()}")


def main():
    parser = argparse.ArgumentParser(description="Smoke test the Cats vs Dogs inference API")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the running API")
    parser.add_argument("--image", required=True, help="Path to a local .jpg/.png to classify")
    args = parser.parse_args()

    try:
        check_health(args.url)
        check_predict_file(args.url, args.image)
        check_predict_base64(args.url, args.image)
    except requests.RequestException as e:
        print(f"[FAIL] Request error: {e}")
        sys.exit(1)
    except AssertionError as e:
        print(f"[FAIL] {e}")
        sys.exit(1)

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
