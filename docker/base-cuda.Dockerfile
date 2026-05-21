# Shared CUDA + PyTorch foundation for transcribe / translate / tts images.
# Build once, then task Dockerfiles FROM video-dubbing-base:local.
#
#   docker build -f docker/base-cuda.Dockerfile -t video-dubbing-base:local .

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.11 && \
    uv sync --no-install-project --no-dev --group cuda-base
