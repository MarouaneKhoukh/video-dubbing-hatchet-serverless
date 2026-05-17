"""
Local pipeline steps — these run on your machine, not on Nebius.
  - download_from_storage / upload_to_storage
  - extract_audio (ffmpeg)
  - translate_text (MADLAD-400, runs on CPU or local GPU)
  - remux_video (ffmpeg)
"""

import logging
import subprocess
from pathlib import Path

import boto3
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from pipeline.config import settings

logger = logging.getLogger(__name__)


# ── Storage ───────────────────────────────────────────────────────────────────

def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=settings.aws_endpoint_url,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )


def download_from_storage(object_key: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading s3://{settings.nebius_bucket_name}/{object_key} → {local_path}")
    _s3_client().download_file(settings.nebius_bucket_name, object_key, str(local_path))


def upload_to_storage(local_path: Path, object_key: str) -> None:
    logger.info(f"Uploading {local_path} → s3://{settings.nebius_bucket_name}/{object_key}")
    _s3_client().upload_file(str(local_path), settings.nebius_bucket_name, object_key)


def object_exists(object_key: str) -> bool:
    try:
        _s3_client().head_object(Bucket=settings.nebius_bucket_name, Key=object_key)
        return True
    except Exception:
        return False


# ── Audio extraction ──────────────────────────────────────────────────────────

def extract_audio(video_path: Path, audio_path: Path) -> None:
    """Extract mono 16kHz WAV for Whisper."""
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000",
        str(audio_path),
    ]
    logger.info(f"Extracting audio: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{result.stderr}")


# ── Translation ───────────────────────────────────────────────────────────────

def _split_text(text: str, max_chars: int = 700) -> list[str]:
    """Split transcript into chunks that fit within the model's context."""
    text = " ".join(text.split())
    chunks, cur = [], ""
    for sent in text.split(". "):
        s = sent.strip()
        if not s:
            continue
        s = s if s.endswith(".") else s + "."
        if len(cur) + len(s) + 1 > max_chars:
            if cur:
                chunks.append(cur.strip())
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        chunks.append(cur)
    return chunks


def translate_text(text: str, target_lang: str) -> str:
    """Translate text using MADLAD-400 (3B)."""
    model_name = "jbochi/madlad400-3b-mt"
    logger.info(f"Loading translation model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    translated_chunks = []
    chunks = _split_text(text)
    logger.info(f"Translating {len(chunks)} chunks → {target_lang}")

    for i, chunk in enumerate(chunks):
        prompt = f"<2{target_lang}> {chunk}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        output = model.generate(**inputs, max_new_tokens=512)
        translated = tokenizer.decode(output[0], skip_special_tokens=True)
        translated_chunks.append(translated)
        logger.info(f"Translated chunk {i + 1}/{len(chunks)}")

    return " ".join(translated_chunks).strip()


# ── Video remux ───────────────────────────────────────────────────────────────

def remux_video(video_path: Path, audio_path: Path, output_path: Path) -> None:
    """Replace original audio track with dubbed audio."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(output_path),
    ]
    logger.info(f"Remuxing video: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg remux failed:\n{result.stderr}")
