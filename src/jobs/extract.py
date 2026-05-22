#!/usr/bin/env python3
"""
Extract audio from videos via ffmpeg subprocess, per file.

Invocation:
    python -m jobs.extract /data/runs/<run_id>/manifests/extract.json
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from pipeline.metadata import (
    load_manifest,
    make_timing,
    manifest_path_from_argv,
    parse_task_runtime,
    record_task_result,
)
from pipeline.paths import build_run_items
from pipeline.storage import data_root
from pipeline.utils import utc_now


def _ffmpeg_extract(video_path: Path, audio_path: Path) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000",
        "-y", str(audio_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _process_file(video_path: Path, audio_path: Path, *, force: bool) -> bool:
    """Returns True if processed, False if skipped (output already exists)."""
    if not force and audio_path.exists():
        print(f"SKIP (already done): {video_path.name}", flush=True)
        return False
    print(f"FILE: {video_path.name} -> {audio_path.name}", flush=True)
    _ffmpeg_extract(video_path, audio_path)
    print(f"  done: {audio_path.name}", flush=True)
    return True


def run_task(config: dict) -> dict:
    """Process all video_keys described by the manifest dict. Writes report; returns payload."""
    started_at = utc_now()
    t0 = time.perf_counter()
    try:
        runtime = parse_task_runtime(config, "extract")
        run_id = runtime["run_id"]
        force = runtime["force"]

        video_keys = config.get("video_keys") or []
        if not video_keys:
            raise ValueError(
                "extract manifest missing 'video_keys'; orchestrator must populate them"
            )

        items = build_run_items(video_keys, run_id)
        data = data_root()

        print(
            f"TASK: extract run_id={run_id} | {len(items)} files | force={force}",
            flush=True,
        )

        processed = 0
        for idx, item in enumerate(items, 1):
            print(f"\n[{idx}/{len(items)}]", flush=True)
            if _process_file(
                data / item["video_key"],
                data / item["audio_key"],
                force=force,
            ):
                processed += 1

        skipped = len(items) - processed
        print(f"\nTask complete: {processed} processed, {skipped} skipped", flush=True)
        result = {
            "audio_keys": [i["audio_key"] for i in items],
            "video_keys": [i["video_key"] for i in items],
            "stems": [i["stem"] for i in items],
            "timing": make_timing(
                "extract", total=len(items), processed=processed, skipped=skipped, t0=t0
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
