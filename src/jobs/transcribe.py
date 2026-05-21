#!/usr/bin/env python3
"""
Transcription + alignment job (ASR + word-level timestamps in one GPU container).

Invocation modes:

  Single-file (local smoke):
      python3 /transcribe.py <audio_wav> <model> <device>

  Run manifest batch:
      python3 /transcribe.py /data/runs/<run_id>/manifests/transcribe.json [chunk_idx]

The manifest describes ``run_id``, task ``config``, and ``input_count``; per-file
paths are derived from upstream stage reports or ``runs/{run_id}/`` artifact layout.
"""

import json
import sys
from pathlib import Path

import whisperx
from faster_whisper import WhisperModel

try:
    import model_cache
except ImportError:
    from models import model_cache

from pipeline.metadata import (
    config_str,
    ensure_torch_device,
    load_manifest,
    parse_manifest_argv,
    parse_task_runtime,
    resolve_chunk,
)

model_cache.configure()
DEFAULTS = model_cache.default_model_spec()

DATA = Path("/data")


def _transcribe_one(
    whisper_model: WhisperModel,
    audio_path: Path,
    transcript_path: Path,
) -> tuple[list[dict], str]:
    raw_segments, info = whisper_model.transcribe(str(audio_path), beam_size=5)
    segments = list(raw_segments)
    print(
        f"  language: {info.language} ({info.language_probability:.2f}) | {len(segments)} segments",
        flush=True,
    )
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(
        "\n".join(s.text.strip() for s in segments if s.text.strip()) + "\n",
        encoding="utf-8",
    )
    wx_segments = [
        {"text": s.text, "start": s.start, "end": s.end}
        for s in segments
        if s.text.strip()
    ]
    return wx_segments, info.language


def _align_one(
    align_cache: dict,
    wx_segments: list[dict],
    audio_path: Path,
    language: str,
    device: str,
    aligned_path: Path,
) -> None:
    if language not in align_cache:
        print(f"  loading align model for language={language}", flush=True)
        model_a, metadata = whisperx.load_align_model(
            language_code=language,
            device=device,
            model_dir=str(model_cache.hf_hub_cache()),
        )
        align_cache[language] = (model_a, metadata)
    model_a, metadata = align_cache[language]
    audio = whisperx.load_audio(str(audio_path))
    result = whisperx.align(wx_segments, model_a, metadata, audio, device)
    aligned_path.parent.mkdir(parents=True, exist_ok=True)
    aligned_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))


def _process_file(
    whisper_model: WhisperModel,
    align_cache: dict,
    device: str,
    audio_path: Path,
    transcript_path: Path,
    aligned_path: Path,
    *,
    force: bool = False,
) -> None:
    if not force and transcript_path.exists() and aligned_path.exists():
        print(f"SKIP (already done): {audio_path.name}", flush=True)
        return
    print(f"FILE: {audio_path.name}", flush=True)
    wx_segments, language = _transcribe_one(whisper_model, audio_path, transcript_path)
    _align_one(align_cache, wx_segments, audio_path, language, device, aligned_path)
    print(f"  done: {transcript_path.name}, {aligned_path.name}", flush=True)


def _load_whisper(model_size: str, device: str, compute_type: str) -> WhisperModel:
    cached = model_cache.whisper_model_cached(model_size)
    if cached:
        print(f"Using cached faster-whisper {model_size}", flush=True)
    else:
        print(f"Downloading faster-whisper {model_size} → {model_cache.faster_whisper_root()}", flush=True)
    return WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        download_root=str(model_cache.faster_whisper_root()),
        local_files_only=cached,
    )


def _run_manifest(manifest_path: Path, chunk_idx: int) -> None:
    manifest = load_manifest(manifest_path)
    runtime = parse_task_runtime(manifest, "transcribe")
    cfg = runtime["config"]
    model_size = config_str(cfg, "model")
    device = ensure_torch_device(config_str(cfg, "device"))
    force = runtime["force"]
    compute_type = "float16" if device == "cuda" else "int8"
    files = resolve_chunk(manifest, chunk_idx)

    print(
        f"MANIFEST: {manifest_path.name} chunk={chunk_idx} | {len(files)} files | "
        f"model={model_size} device={device} force={force}",
        flush=True,
    )
    print(f"Loading model {model_size} on {device}", flush=True)
    whisper_model = _load_whisper(model_size, device, compute_type)
    align_cache: dict = {}

    for idx, item in enumerate(files, 1):
        print(f"\n[{idx}/{len(files)}]", flush=True)
        _process_file(
            whisper_model,
            align_cache,
            device,
            DATA / item["audio_key"],
            DATA / item["transcript_key"],
            DATA / item["aligned_key"],
            force=force,
        )

    print(f"\nChunk complete: {len(files)} files processed", flush=True)


def _run_single(audio_path: Path, model_size: str, device: str) -> None:
    compute_type = "float16" if device == "cuda" else "int8"
    transcript_path = audio_path.with_suffix(".txt")
    aligned_path = audio_path.parent / (audio_path.stem + "_aligned.json")

    print(f"Loading model {model_size} on {device}", flush=True)
    whisper_model = _load_whisper(model_size, device, compute_type)
    _process_file(whisper_model, {}, device, audio_path, transcript_path, aligned_path)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: transcribe.py <audio_wav> <model> <device>  OR  "
            "transcribe.py <task_manifest.json> [chunk_idx]"
        )
    arg1 = Path(sys.argv[1])
    if arg1.suffix == ".json":
        manifest_path, chunk_idx = parse_manifest_argv()
        _run_manifest(manifest_path, chunk_idx)
        return
    model_size = sys.argv[2] if len(sys.argv) > 2 else DEFAULTS.transcribe_model
    device = sys.argv[3] if len(sys.argv) > 3 else DEFAULTS.transcribe_device
    _run_single(arg1, model_size, device)


if __name__ == "__main__":
    main()
