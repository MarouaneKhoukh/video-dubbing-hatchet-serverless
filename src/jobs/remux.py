#!/usr/bin/env python3
"""
Remux dubbed audio onto original video via ffmpeg subprocess, per file.

Invocation:
    python -m jobs.remux /data/runs/<run_id>/manifests/remux.json
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
    resolve_manifest_stems,
    video_keys_by_stem,
)
from pipeline.paths import build_run_items_from_stems
from pipeline.storage import auto_configure_data_root, data_root, staged_write
from pipeline.utils import utc_now

# Wire the /data FUSE mount into pipeline.storage so manifest reads use the
# bucket directly instead of an S3 client (no AWS creds inside the container).
auto_configure_data_root()


def _ffmpeg_remux(video_path: Path, dubbed_path: Path, output_path: Path) -> None:
    # Kokoro emits 24 kHz mono WAV with channel_layout="unknown", which makes
    # some players (notably QuickTime / Apple Preview / VSCode preview) skip
    # the audio track silently after remux. Fix with an explicit filter chain:
    #   - aformat=channel_layouts=mono   → stamp the input as proper mono
    #   - pan=stereo|c0=c0|c1=c0         → duplicate mono into both channels
    #   - aresample=48000                → resample to standard 48 kHz
    # Then encode as AAC LC at 128 kbps — the universal MP4 audio baseline.
    # Stage the MP4 in /tmp because the moov atom needs seek-back finalization,
    # and FUSE-mounted /data can't seek.
    with staged_write(output_path) as out_path:
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-i", str(dubbed_path),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy",
            "-af", "aformat=channel_layouts=mono,pan=stereo|c0=c0|c1=c0,aresample=48000",
            "-c:a", "aac", "-profile:a", "aac_low", "-b:a", "128k",
            "-shortest", "-movflags", "+faststart",
            "-y", str(out_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"ffmpeg remux failed for {video_path.name} "
                f"(exit {e.returncode}): {e.stderr.strip()[-800:]}"
            ) from e


def _process_file(
    video_path: Path,
    dubbed_path: Path,
    output_path: Path,
    *,
    force: bool,
) -> bool:
    """Returns True if processed, False if skipped."""
    if not force and output_path.exists():
        print(f"SKIP (already done): {output_path.name}", flush=True)
        return False
    print(f"FILE: {video_path.name} + {dubbed_path.name} -> {output_path.name}", flush=True)
    _ffmpeg_remux(video_path, dubbed_path, output_path)
    print(f"  done: {output_path.name}", flush=True)
    return True


def run_task(config: dict) -> dict:
    """Process all files described by the manifest dict. Writes report; returns payload."""
    started_at = utc_now()
    t0 = time.perf_counter()
    try:
        runtime = parse_task_runtime(config, "remux")
        run_id = runtime["run_id"]
        force = runtime["force"]

        stems = resolve_manifest_stems(config)
        items = build_run_items_from_stems(
            stems,
            run_id,
            video_keys_by_stem=video_keys_by_stem(run_id),
        )

        missing_video = [i["stem"] for i in items if not i["video_key"]]
        if missing_video:
            raise RuntimeError(
                f"[remux] missing original video_key for stems: {missing_video[:5]}. "
                f"Re-run extract or check runs/{run_id}/reports/extract.json"
            )

        print(
            f"TASK: remux run_id={run_id} | {len(items)} files | force={force}",
            flush=True,
        )

        data = data_root()
        processed = 0
        for idx, item in enumerate(items, 1):
            print(f"\n[{idx}/{len(items)}]", flush=True)
            if _process_file(
                data / item["video_key"],
                data / item["dubbed_key"],
                data / item["output_key"],
                force=force,
            ):
                processed += 1

        skipped = len(items) - processed
        print(f"\nTask complete: {processed} processed, {skipped} skipped", flush=True)
        result = {
            "output_keys": [i["output_key"] for i in items],
            "count": len(items),
            "timing": make_timing(
                "remux", total=len(items), processed=processed, skipped=skipped, t0=t0
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
