"""
FastAPI inference service for the Cats vs Dogs classifier (M2).

Endpoints:
    GET  /health          - liveness/readiness probe; reports whether the
                             trained model is loaded and which device it's on
    POST /predict          - multipart/form-data image upload -> prediction
    POST /predict/base64   - JSON {"image_base64": "..."} -> prediction

Run locally:
    uvicorn app:app --app-dir src --reload --host 0.0.0.0 --port 8000

Then open http://localhost:8000/docs for interactive Swagger UI, or see
README.md for curl examples. Runs the same way inside the Docker container
(see project Dockerfile) - only the host changes.
"""
import base64
import binascii
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

# Quiet down transformers' advisory logging (see train.py for the full
# explanation) - harmless either way, but keeps container logs clean.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
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

# Model + transform are loaded once at startup and reused across requests -
# reloading per-request would be slow and pointless since weights don't change.
state = {"model": None, "device": None, "eval_transform": None, "image_size": 224}


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
    }


def _predict_image(img: Image.Image) -> PredictResponse:
    """Shared prediction logic for both /predict and /predict/base64."""
    if state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model not loaded - expected weights at {MODEL_PATH}. "
                f"Train the model first (python src/train.py) and ensure "
                f"models/model.pt is present, then restart the service."
            ),
        )
    img = img.convert("RGB")
    tensor = state["eval_transform"](img)
    result = run_predict(state["model"], tensor, device=state["device"], class_names=CLASS_NAMES)
    probabilities = {cls: round(p, 4) for cls, p in zip(CLASS_NAMES, result["probabilities"])}
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
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    return _predict_image(img)


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
        raise HTTPException(
            status_code=400,
            detail="image_base64 is not a valid base64-encoded image.",
        )
    return _predict_image(img)
