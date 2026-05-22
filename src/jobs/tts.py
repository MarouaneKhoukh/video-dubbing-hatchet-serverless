#!/usr/bin/env python3
"""
Kokoro TTS job.

Invocation:
    python -m jobs.tts /data/runs/<run_id>/manifests/tts.json
"""

import time
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
    make_timing,
    manifest_path_from_argv,
    parse_task_runtime,
    record_task_result,
    resolve_manifest_stems,
)
from pipeline.paths import build_run_items_from_stems
from pipeline.storage import data_root
from pipeline.utils import utc_now

model_cache.configure()


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
) -> bool:
    """Returns True if processed, False if skipped."""
    if not force and output_path.exists():
        print(f"SKIP (already done): {input_path.name}", flush=True)
        return False
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
    return True


def _load_pipeline(lang: str, repo: str, device: str) -> KPipeline:
    ensure_torch_device(device)
    if model_cache.hf_model_cached(repo):
        print(f"Using cached Kokoro ({repo})", flush=True)
    else:
        print(f"Downloading Kokoro ({repo})", flush=True)
    print(f"Loading Kokoro on {device} (lang={lang})", flush=True)
    return KPipeline(lang_code=lang, repo_id=repo, device=device)


def run_task(config: dict) -> dict:
    """Process all files described by the manifest dict. Writes report; returns payload."""
    started_at = utc_now()
    t0 = time.perf_counter()
    try:
        model_cache.configure()
        runtime = parse_task_runtime(config, "tts")
        cfg = runtime["config"]
        run_id = runtime["run_id"]
        voice = config_str(cfg, "voice")
        lang = config_str(cfg, "lang")
        repo = config_str(cfg, "repo")
        device = ensure_torch_device(config_str(cfg, "device"))
        force = runtime["force"]

        stems = resolve_manifest_stems(config)
        items = build_run_items_from_stems(stems, run_id)

        print(
            f"TASK: tts run_id={run_id} | {len(items)} files | "
            f"voice={voice} lang={lang} device={device} force={force}",
            flush=True,
        )
        pipeline = _load_pipeline(lang, repo, device)

        data = data_root()
        processed = 0
        for idx, item in enumerate(items, 1):
            print(f"\n[{idx}/{len(items)}]", flush=True)
            if _synthesize_one(
                pipeline, voice,
                data / item["translated_key"],
                data / item["dubbed_key"],
                force=force,
            ):
                processed += 1

        skipped = len(items) - processed
        print(f"\nTask complete: {processed} processed, {skipped} skipped", flush=True)
        result = {
            "dubbed_keys": [i["dubbed_key"] for i in items],
            "timing": make_timing(
                "tts", total=len(items), processed=processed, skipped=skipped, t0=t0
            ),
        }
        record_task_result(config, result, started_at=started_at)
        return result
    except Exception as exc:
        record_task_result(config, {}, started_at=started_at, failed=True, error=str(exc))
        raise


def main() -> None:
    config = load_manifest(manifest_path_from_argv())
    run_task(config)


if __name__ == "__main__":
    main()
