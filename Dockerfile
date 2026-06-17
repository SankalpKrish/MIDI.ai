FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TF_FORCE_GPU_ALLOW_GROWTH=true

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "\
from demucs.pretrained import get_model; get_model('htdemucs'); \
import tensorflow_hub; tensorflow_hub.load('https://tfhub.dev/google/yamnet/1'); \
"

COPY . .

ENTRYPOINT ["python", "audio_pipeline.py"]
