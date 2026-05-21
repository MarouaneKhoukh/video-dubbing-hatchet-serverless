#!/usr/bin/env python3
"""
Kokoro TTS job.

Invocation modes:

  Single-file (legacy):
      python3 /tts.py <input_txt> <output_wav> [voice] [lang_code]

  Run manifest batch:
      python3 /tts.py /data/runs/<run_id>/manifests/tts.json [chunk_idx]
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro import KPipeline

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


def _split_text(text: str, max_chars: int = 500) -> list[str]:
    text = " ".join(text.split())
    chunks: list[str] = []
    cur = ""
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


def _synthesize_one(
    pipeline: KPipeline,
    voice: str,
    input_path: Path,
    output_path: Path,
    *,
    force: bool = False,
) -> None:
    if not force and output_path.exists():
        print(f"SKIP (already done): {input_path.name}", flush=True)
        return
    print(f"FILE: {input_path.name} -> {output_path.name}", flush=True)
    text = input_path.read_text(encoding="utf-8").strip()
    chunks = _split_text(text)
    print(f"  chunks: {len(chunks)}", flush=True)

    audio_chunks: list[np.ndarray] = []
    for i, chunk in enumerate(chunks):
        for _, _, audio in pipeline(chunk, voice=voice, speed=1.0):
            audio_chunks.append(audio)
        print(f"  chunk {i + 1}/{len(chunks)} done", flush=True)

    if not audio_chunks:
        raise RuntimeError(f"No audio generated for {input_path.name} — check input text")

    combined = np.concatenate(audio_chunks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), combined, 24000)


def _load_pipeline(lang: str, repo: str, device: str) -> KPipeline:
    ensure_torch_device(device)
    if model_cache.hf_model_cached(repo):
        print(f"Using cached Kokoro ({repo})", flush=True)
    else:
        print(f"Downloading Kokoro ({repo})", flush=True)
    print(f"Loading Kokoro on {device} (lang={lang})", flush=True)
    return KPipeline(lang_code=lang, repo_id=repo, device=device)


def _run_manifest(manifest_path: Path, chunk_idx: int) -> None:
    manifest = load_manifest(manifest_path)
    runtime = parse_task_runtime(manifest, "tts")
    cfg = runtime["config"]
    voice = config_str(cfg, "voice")
    lang = config_str(cfg, "lang")
    repo = config_str(cfg, "repo")
    device = ensure_torch_device(config_str(cfg, "device"))
    force = runtime["force"]
    files = resolve_chunk(manifest, chunk_idx)

    print(
        f"MANIFEST: {manifest_path.name} chunk={chunk_idx} | {len(files)} files | "
        f"voice={voice} lang={lang} device={device} force={force}",
        flush=True,
    )
    pipeline = _load_pipeline(lang, repo, device)

    for idx, item in enumerate(files, 1):
        print(f"\n[{idx}/{len(files)}]", flush=True)
        _synthesize_one(
            pipeline, voice,
            DATA / item["translated_key"],
            DATA / item["dubbed_key"],
            force=force,
        )

    print(f"\nChunk complete: {len(files)} files processed", flush=True)


def _run_single(
    input_path: Path,
    output_path: Path,
    voice: str,
    lang: str,
    *,
    device: str = "cpu",
) -> None:
    repo = DEFAULTS.tts_repo
    print(f"Loading Kokoro pipeline (voice={voice}, lang={lang}, device={device})", flush=True)
    pipeline = _load_pipeline(lang, repo, device)
    _synthesize_one(pipeline, voice, input_path, output_path)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: tts.py <input_txt> <output_wav> [voice] [lang]  OR  "
            "tts.py <task_manifest.json> [chunk_idx]"
        )
    arg1 = Path(sys.argv[1])
    if arg1.suffix == ".json":
        manifest_path, chunk_idx = parse_manifest_argv()
        _run_manifest(manifest_path, chunk_idx)
        return
    output_path = Path(sys.argv[2])
    voice = sys.argv[3] if len(sys.argv) > 3 else DEFAULTS.tts_voice
    lang = sys.argv[4] if len(sys.argv) > 4 else DEFAULTS.tts_lang
    _run_single(arg1, output_path, voice, lang)


if __name__ == "__main__":
    main()
