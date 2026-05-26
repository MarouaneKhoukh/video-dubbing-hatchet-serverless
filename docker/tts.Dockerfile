# syntax=docker/dockerfile:1
ARG BASE_IMAGE=video-dubbing-base:local
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# Layer 3 — task-specific system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Layer 4 — lockfiles
COPY pyproject.toml ./

# Layer 5 — task Python deps
ARG USE_CUDA_SOURCES=0
COPY scripts/uv_sync_task.sh /usr/local/bin/uv-sync-task
RUN chmod +x /usr/local/bin/uv-sync-task
RUN --mount=type=cache,target=/root/.cache/uv \
    USE_CUDA_SOURCES=${USE_CUDA_SOURCES} uv-sync-task tts

# Layer 6 — source packages
COPY src/pipeline /pipeline
COPY src/jobs     /jobs
COPY src/models   /models

ENV PYTHONPATH=/ \
    MODEL_CACHE_DIR=/data/models \
    HF_HOME=/data/models/huggingface \
    HUGGINGFACE_HUB_CACHE=/data/models/huggingface/hub \
    TORCH_HOME=/data/models/torch \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

ENTRYPOINT ["python3", "-m", "jobs.tts"]
