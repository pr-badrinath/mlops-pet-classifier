"""
FastAPI inference service for the Cats vs Dogs classifier (M2 + M5).

Endpoints:
    GET  /health          - liveness/readiness probe; reports whether the
                             trained model is loaded and which device it's on
    GET  /metrics          - basic in-app monitoring: request counts, error
                             counts, and latency stats per endpoint (M5)
    POST /predict          - multipart/form-data image upload -> prediction
    POST /predict/base64   - JSON {"image_base64": "..."} -> prediction

Monitoring & logging (M5):
    Every request gets a structured JSON access-log line (method, path,
    status, latency) via middleware. Every successful prediction additionally
    gets a JSON "prediction" log line (input metadata + predicted label +
    probabilities + inference latency) - deliberately never logs the raw
    image bytes or base64 string itself, only metadata about them, per the
    assignment's "excluding sensitive data" requirement. Logs go to stdout
    (so `docker compose logs` / `docker logs` show them) and to rotating
    files under LOG_DIR (default "logs/", see docker-compose.yml for the
    volume mount that persists these outside the container).

Run locally:
    uvicorn app:app --app-dir src --reload --host 0.0.0.0 --port 8000

Then open http://localhost:8000/docs for interactive Swagger UI, or see
README.md for curl examples. Runs the same way inside the Docker container
(see project Dockerfile) - only the host changes.
"""
import base64
import binascii
import io
import json
import logging
import logging.handlers
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

# Quiet down transformers' advisory logging (see train.py for the full
# explanation) - harmless either way, but keeps container logs clean.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from dataset import get_transforms
from model import SimpleCNN, predict as run_predict
from utils import get_device, load_params

CLASS_NAMES = ["Cat", "Dog"]

# Overridable via environment variables so the same image works whether
# you're running locally (`models/model.pt` relative to project root) or
# inside the container (paths are set explicitly in the Dockerfile COPY).
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/model.pt"))
PARAMS_PATH = Path(os.environ.get("PARAMS_PATH", "params.yaml"))
LOG_DIR = Path(os.environ.get("LOG_DIR", "logs"))

# Model + transform are loaded once at startup and reused across requests -
# reloading per-request would be slow and pointless since weights don't change.
state = {"model": None, "device": None, "eval_transform": None, "image_size": 224}


# --------------------------------------------------------------------------
# M5: structured JSON logging
# --------------------------------------------------------------------------
class JsonFormatter(logging.Formatter):
    """Renders each log record as a single JSON line - easy to grep, ship
    to a log aggregator, or parse later, and keeps every field explicit
    (so it's obvious at a glance that no raw image data ever gets logged).
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
        }
        # Loggers below are always called with a dict payload (see calls
        # further down), not a plain string - this keeps fields structured.
        if isinstance(record.msg, dict):
            payload.update(record.msg)
        else:
            payload["message"] = record.getMessage()
        return json.dumps(payload, default=str)


def _configure_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:  # avoid duplicate handlers on uvicorn --reload
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(JsonFormatter())
        logger.addHandler(stream_handler)

        file_handler = logging.handlers.RotatingFileHandler(
            LOG_DIR / f"{name}.log", maxBytes=5_000_000, backupCount=3
        )
        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)
    return logger


access_logger = _configure_logger("access")
prediction_logger = _configure_logger("prediction")


# --------------------------------------------------------------------------
# M5: basic in-app metrics (request count + latency per endpoint)
# --------------------------------------------------------------------------
_metrics_lock = threading.Lock()
_metrics_store: dict = {}
_service_start_time = time.time()


def _record_metric(path: str, latency_ms: float, status_code: int) -> None:
    with _metrics_lock:
        entry = _metrics_store.setdefault(
            path,
            {"count": 0, "error_count": 0, "total_latency_ms": 0.0,
             "min_latency_ms": None, "max_latency_ms": None},
        )
        entry["count"] += 1
        if status_code >= 400:
            entry["error_count"] += 1
        entry["total_latency_ms"] += latency_ms
        entry["min_latency_ms"] = latency_ms if entry["min_latency_ms"] is None else min(entry["min_latency_ms"], latency_ms)
        entry["max_latency_ms"] = latency_ms if entry["max_latency_ms"] is None else max(entry["max_latency_ms"], latency_ms)


def _load_model_if_available() -> None:
    """Load the trained model into `state`, or leave it as None if missing.

    Deliberately non-fatal: the service should still start (and /health
    should still respond, just reporting model_loaded=False) even if you
    haven't trained/copied a model in yet - this matters for container
    startup ordering and for smoke-testing the service shape early.
    """
    device = get_device()
    state["device"] = device

    try:
        cfg = load_params(str(PARAMS_PATH))
        state["image_size"] = cfg["data"]["image_size"]
    except Exception:
        pass  # fall back to the default image_size already in `state`

    _, eval_transform = get_transforms(state["image_size"])
    state["eval_transform"] = eval_transform

    if MODEL_PATH.exists():
        model = SimpleCNN(num_classes=len(CLASS_NAMES))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        state["model"] = model
        print(f"[startup] Loaded model from {MODEL_PATH} onto {device}")
    else:
        state["model"] = None
        print(f"[startup] WARNING: no model found at {MODEL_PATH}. "
              f"/predict will return 503 until one is trained and placed there.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model_if_available()
    yield


app = FastAPI(
    title="Cats vs Dogs Classifier API",
    description="Binary image classification (Cat/Dog) inference service for a pet-adoption platform.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def logging_and_metrics_middleware(request: Request, call_next):
    """Access-log + metrics for every request, regardless of endpoint.

    Deliberately only touches request metadata (method, path, client host)
    - never the request/response body - so this stays cheap and side-effect
    free even for large file uploads, and never risks logging image data.
    """
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    _record_metric(request.url.path, latency_ms, response.status_code)
    access_logger.info({
        "event": "http_request",
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "latency_ms": latency_ms,
        "client_ip": request.client.host if request.client else None,
    })
    response.headers["X-Process-Time-Ms"] = str(latency_ms)
    return response


class Base64PredictRequest(BaseModel):
    image_base64: str


class PredictResponse(BaseModel):
    label: str
    class_index: int
    probabilities: dict


@app.get("/health")
def health():
    """Liveness/readiness probe. Reports whether a trained model is loaded."""
    return {
        "status": "ok",
        "model_loaded": state["model"] is not None,
        "device": str(state["device"]) if state["device"] else None,
        "uptime_seconds": round(time.time() - _service_start_time, 1),
    }


@app.get("/metrics")
def metrics():
    """Basic in-app monitoring (M5): request count, error count, and
    latency stats per endpoint, plus overall service uptime. Resets on
    restart - this is intentionally simple in-process state (the
    assignment explicitly allows "logs, Prometheus, or simple in-app
    counters"; this is the counters option). For durable/cross-instance
    metrics in a real deployment, ship these numbers to Prometheus instead.
    """
    with _metrics_lock:
        snapshot = {}
        for path, entry in _metrics_store.items():
            avg = entry["total_latency_ms"] / entry["count"] if entry["count"] else 0
            snapshot[path] = {
                "request_count": entry["count"],
                "error_count": entry["error_count"],
                "avg_latency_ms": round(avg, 2),
                "min_latency_ms": entry["min_latency_ms"],
                "max_latency_ms": entry["max_latency_ms"],
            }
    return {
        "uptime_seconds": round(time.time() - _service_start_time, 1),
        "endpoints": snapshot,
    }


def _predict_image(img: Image.Image, source_meta: dict) -> PredictResponse:
    """Shared prediction logic for both /predict and /predict/base64.

    `source_meta` carries only non-sensitive metadata about the input
    (e.g. filename, content-type, byte size) for the prediction log line -
    never the image bytes themselves.
    """
    if state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model not loaded - expected weights at {MODEL_PATH}. "
                f"Train the model first (python src/train.py) and ensure "
                f"models/model.pt is present, then restart the service."
            ),
        )
    start = time.perf_counter()
    img = img.convert("RGB")
    tensor = state["eval_transform"](img)
    result = run_predict(state["model"], tensor, device=state["device"], class_names=CLASS_NAMES)
    inference_ms = round((time.perf_counter() - start) * 1000, 2)

    probabilities = {cls: round(p, 4) for cls, p in zip(CLASS_NAMES, result["probabilities"])}

    prediction_logger.info({
        "event": "prediction",
        **source_meta,
        "predicted_label": result["label"],
        "probabilities": probabilities,
        "inference_ms": inference_ms,
    })

    return PredictResponse(
        label=result["label"],
        class_index=result["class_index"],
        probabilities=probabilities,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict_file(file: UploadFile = File(...)):
    """Predict from an uploaded image file (multipart/form-data).

    Example (curl):
        curl -X POST http://localhost:8000/predict -F "file=@/path/to/pet.jpg"
    """
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents))
        img.load()
    except (UnidentifiedImageError, OSError):
        prediction_logger.warning({"event": "invalid_image", "source": "file_upload", "filename": file.filename})
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    source_meta = {
        "source": "file_upload",
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(contents),
    }
    return _predict_image(img, source_meta)


@app.post("/predict/base64", response_model=PredictResponse)
def predict_base64(payload: Base64PredictRequest):
    """Predict from a base64-encoded image string (JSON body).

    Example (curl, PowerShell-friendly):
        $b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("pet.jpg"))
        curl -X POST http://localhost:8000/predict/base64 `
             -H "Content-Type: application/json" `
             -d (@{image_base64=$b64} | ConvertTo-Json)
    """
    try:
        image_bytes = base64.b64decode(payload.image_base64, validate=True)
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except (UnidentifiedImageError, OSError, ValueError, binascii.Error):
        prediction_logger.warning({"event": "invalid_image", "source": "base64_json"})
        raise HTTPException(
            status_code=400,
            detail="image_base64 is not a valid base64-encoded image.",
        )
    source_meta = {"source": "base64_json", "size_bytes": len(image_bytes)}
    return _predict_image(img, source_meta)
