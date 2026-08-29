# Cats vs Dogs classifier - inference service image.
# Build:  docker build -t pet-classifier-api:latest .
# Run:    docker run -p 8000:8000 pet-classifier-api:latest
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_VERBOSITY=error \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1

WORKDIR /app

# Install dependencies first and separately from app code so Docker's layer
# cache is reused across builds unless requirements.txt actually changes -
# avoids reinstalling torch (the slowest part) on every code edit.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code, config, and the trained model. model.pt must already exist
# locally (run `python src/train.py` first) - this build does not train.
COPY src/ ./src/
COPY params.yaml .
COPY models/model.pt ./models/model.pt

EXPOSE 8000

# Container-level health check hitting our own /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
