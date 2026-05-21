#!/usr/bin/env sh
# Install one task group on top of video-dubbing-base:local (pyproject.toml only, no lockfile).
# USE_CUDA_SOURCES=1 → Nebius / Linux+NVIDIA (cu124 torch from cuda-base)
# USE_CUDA_SOURCES=0 → Mac / CPU (PyPI torch; default)
set -e
GROUP="$1"
if [ "$USE_CUDA_SOURCES" = "1" ]; then
  uv sync --no-install-project --no-dev --group "$GROUP"
else
  uv sync --no-install-project --no-dev --group cpu-base --group "$GROUP" --no-sources
  # whisperx pulls CUDA torchaudio unless we pin CPU wheels after the full dep tree is installed
  uv pip install --no-deps --force-reinstall "torch==2.5.1" "torchaudio==2.5.1"
fi
