# syntax=docker/dockerfile:1
# Extract audio from video files. CPU only, ffmpeg + lightweight Python deps.
ARG BASE_IMAGE=video-dubbing-base:local
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# Layer 3 — task-specific system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Layer 4 — source packages (no extra Python deps beyond base; uses pipeline.* + boto3 already there)
COPY src/pipeline /pipeline
COPY src/jobs     /jobs

ENV PYTHONPATH=/

ENTRYPOINT ["python3", "-m", "jobs.extract"]
