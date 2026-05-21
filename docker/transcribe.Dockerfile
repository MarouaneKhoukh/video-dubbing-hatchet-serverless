# syntax=docker/dockerfile:1
ARG BASE_IMAGE=video-dubbing-base:local
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# Layer 3 — task-specific system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Layer 4 — deps manifest (re-copy when groups change; base may already have pyproject.toml)
COPY pyproject.toml ./

# Layer 5 — task Python deps (incremental on base .venv)
ARG USE_CUDA_SOURCES=0
COPY scripts/uv_sync_task.sh /usr/local/bin/uv-sync-task
RUN chmod +x /usr/local/bin/uv-sync-task
RUN --mount=type=cache,target=/root/.cache/uv \
    USE_CUDA_SOURCES=${USE_CUDA_SOURCES} uv-sync-task transcribe

# Layer 6 — job script (changes most often)
COPY src/pipeline /pipeline
COPY src/models/whisper.py /whisper.py
COPY src/models/model_cache.py /model_cache.py
COPY src/jobs/transcribe.py /transcribe.py
COPY src/models/download_models.py /download_models.py

ENV PYTHONPATH=/ \
    MODEL_CACHE_DIR=/data/models \
    HF_HOME=/data/models/huggingface \
    HUGGINGFACE_HUB_CACHE=/data/models/huggingface/hub \
    TORCH_HOME=/data/models/torch

ENTRYPOINT ["python3", "/transcribe.py"]
