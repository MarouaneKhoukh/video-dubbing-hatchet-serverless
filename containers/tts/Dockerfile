FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip espeak-ng libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121 && \
    python3 -m pip install --no-cache-dir TTS

COPY synthesize.py /synthesize.py

ENTRYPOINT ["python3", "/synthesize.py"]
