# syntax=docker/dockerfile:1
# Remux dubbed audio onto original video. CPU only, ffmpeg + lightweight Python deps.
ARG BASE_IMAGE=video-dubbing-base:local
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# Layer 3 — task-specific system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Layer 4 — source packages
COPY src/pipeline /pipeline
COPY src/jobs     /jobs

ENV PYTHONPATH=/

ENTRYPOINT ["python3", "-m", "jobs.remux"]
