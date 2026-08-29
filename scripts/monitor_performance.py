"""
Post-deployment model performance tracking (M5).

Sends a small batch of images with KNOWN true labels to a *running*
inference service (local uvicorn, `docker compose up`, or the CD-deployed
container), collects predictions, and reports accuracy/precision/recall.
This is the M5 requirement to "collect a small batch of real or simulated
requests and true labels" and track model performance post-deployment -
distinct from M1's MLflow metrics, which only cover training-time evaluation.

Usage:
    python scripts/monitor_performance.py
    python scripts/monitor_performance.py --url http://localhost:8000 --samples-per-class 10
    python scripts/monitor_performance.py --data-dir data/raw/PetImages --output-dir monitoring/reports

Requires a folder with Cat/ and Dog/ subfolders of labeled images
(defaults to data/raw/PetImages, the same layout used since M1).
"""
import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CLASSES = ["Cat", "Dog"]


def sample_images(data_dir: Path, samples_per_class: int, seed: int = 42) -> list:
    """Return [(path, true_label), ...] sampled from data_dir/Cat and data_dir/Dog."""
    rng = random.Random(seed)
    batch = []
    for cls in CLASSES:
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            print(f"[WARN] {cls_dir} not found - skipping class '{cls}'")
            continue
        files = [p for p in cls_dir.iterdir() if p.is_file()]
        if not files:
            print(f"[WARN] no files found in {cls_dir}")
            continue
        chosen = rng.sample(files, min(samples_per_class, len(files)))
        batch.extend((f, cls) for f in chosen)
    rng.shuffle(batch)
    return batch


def call_predict(url: str, image_path: Path) -> dict:
    """POST one image to /predict and return the parsed response + latency."""
    start = time.perf_counter()
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{url}/predict",
            files={"file": (image_path.name, f, "image/jpeg")},
            timeout=30,
        )
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    resp.raise_for_status()
    body = resp.json()
    body["latency_ms"] = latency_ms
    return body


def main():
    parser = argparse.ArgumentParser(description="Post-deployment model performance tracking")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the running API")
    parser.add_argument("--data-dir", default="data/raw/PetImages",
                         help="Folder with Cat/ and Dog/ subfolders of labeled images")
    parser.add_argument("--samples-per-class", type=int, default=5, help="How many images per class to sample")
    parser.add_argument("--output-dir", default="monitoring/reports", help="Where to save the JSON report")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    batch = sample_images(data_dir, args.samples_per_class, args.seed)
    if not batch:
        print(f"[FAIL] No labeled images found under {data_dir}. "
              f"Point --data-dir at a folder with Cat/ and Dog/ subfolders.")
        sys.exit(1)

    print(f"Sending {len(batch)} labeled requests to {args.url}/predict ...\n")
    y_true, y_pred, records = [], [], []

    for path, true_label in batch:
        try:
            result = call_predict(args.url, path)
        except requests.RequestException as e:
            print(f"[FAIL] Request error on {path.name}: {e}")
            sys.exit(1)

        y_true.append(true_label)
        y_pred.append(result["label"])
        records.append({
            "file": path.name,
            "true_label": true_label,
            "predicted_label": result["label"],
            "correct": true_label == result["label"],
            "probabilities": result["probabilities"],
            "latency_ms": result["latency_ms"],
        })
        status = "OK  " if true_label == result["label"] else "MISS"
        print(f"  [{status}] {path.name:20s} true={true_label:4s} pred={result['label']:4s} "
              f"({result['latency_ms']:.1f} ms)")

    acc = accuracy_score(y_true, y_pred)
    report_text = classification_report(y_true, y_pred, labels=CLASSES, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES).tolist()
    avg_latency = sum(r["latency_ms"] for r in records) / len(records)

    print(f"\n=== Post-deployment performance report ===")
    print(f"Batch size:   {len(records)}")
    print(f"Accuracy:     {acc:.4f}")
    print(f"Avg latency:  {avg_latency:.1f} ms")
    print(report_text)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "api_url": args.url,
        "batch_size": len(records),
        "accuracy": acc,
        "avg_latency_ms": round(avg_latency, 2),
        "confusion_matrix": {"labels": CLASSES, "matrix": cm},
        "classification_report": report_text,
        "records": records,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"perf_report_{ts}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Saved report to {out_path}")


if __name__ == "__main__":
    main()
