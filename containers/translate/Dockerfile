FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    transformers>=4.40.0 \
    torch \
    sentencepiece \
    accelerate

COPY translate.py /translate.py

ENTRYPOINT ["python3", "/translate.py"]
