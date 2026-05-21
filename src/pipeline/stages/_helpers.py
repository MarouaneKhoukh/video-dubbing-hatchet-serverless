"""Shared helpers for pipeline stage runners."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from pipeline.batch_runner import TaskTiming
from pipeline.metadata import resolve_upstream_stems, video_keys_by_stem
from pipeline.paths import build_run_items, build_run_items_from_stems
from pipeline.run import PipelineRun


def stage_items(run: PipelineRun, stage: str) -> list[dict[str, str]]:
    """Items for *stage* — from upstream artifacts, or ``run.video_keys`` if not yet materialized."""
    if stage == "extract":
        return build_run_items(run.video_keys, run.run_id)
    stems = resolve_upstream_stems(run.run_id, stage)
    if stems:
        return build_run_items_from_stems(
            stems,
            run.run_id,
            video_keys_by_stem=video_keys_by_stem(run.run_id),
        )
    return build_run_items(run.video_keys, run.run_id)


def require_inputs(
    items: list[dict[str, str]],
    *,
    stage: str,
    upstream: str,
    keys: str | tuple[str, ...],
) -> None:
    """Fail fast when upstream artifacts are missing (e.g. transcribe without extract)."""
    from pipeline.storage import object_exists

    key_list = (keys,) if isinstance(keys, str) else keys
    missing: list[dict[str, str]] = []
    for item in items:
        if not all(object_exists(item[k]) for k in key_list):
            missing.append(item)

    if not missing:
        return

    examples: list[str] = []
    for item in missing[:3]:
        for key in key_list:
            path = item[key]
            if not object_exists(path):
                examples.append(path)
    sample = ", ".join(examples[:5]) + ("…" if len(examples) > 5 else "")
    raise RuntimeError(
        f"[{stage}] {len(missing)}/{len(items)} item(s) missing {upstream} input(s). "
        f"Run the `{upstream}` stage first (or use a fresh `--run-id` if inputs changed). "
        f"Missing: {sample}"
    )


def timing_payload(timing: TaskTiming) -> dict[str, Any]:
    d = asdict(timing)
    d["wall_s"] = round(timing.wall_s, 2)
    d["per_file_s"] = (
        round(timing.wall_s / timing.processed_files, 2) if timing.processed_files else 0.0
    )
    for chunk in d["chunks"]:
        chunk["wall_s"] = round(chunk["wall_s"], 2)
    return d

