"""Sequential pipeline runner — Python in-process orchestrator.

Writes the task manifest, imports the matching ``jobs/<stage>.py``, calls
``run_task(config)`` in-process. The job writes its own report (success or
failure); the orchestrator only writes a fallback failure report if the import
itself or the manifest write blows up before the job's own try/except runs.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable

from pydantic import BaseModel, Field

from pipeline.config import get_config

logger = logging.getLogger(__name__)

STAGE_ORDER: tuple[str, ...] = ("extract", "transcribe", "translate", "tts", "remux")


class PipelineRun(BaseModel):
    """One dubbing run: one or many input videos under the bucket /data mount."""

    video_keys: list[str]
    target_lang: str = Field(default_factory=lambda: get_config().pipeline.target_lang)
    run_id: str = "demo"
    batch_id: str = "default"
    force: bool = False


def _job_module(stage: str):
    if stage not in STAGE_ORDER:
        known = ", ".join(STAGE_ORDER)
        raise ValueError(f"Unknown stage {stage!r}; expected one of: {known}")
    return importlib.import_module(f"jobs.{stage}")


def run_stage(
    stage: str,
    run: PipelineRun,
    *,
    cli_overrides: dict[str, dict] | None = None,
    log: Callable[[str], None] = logger.info,
) -> dict:
    """Run one pipeline stage in-process. Returns the report payload."""
    from pipeline.metadata import build_task_manifest
    from pipeline.paths import task_manifest_object_key
    from pipeline.storage import upload_json

    log(f"=== {stage} ===")
    manifest = build_task_manifest(
        run, stage, cli_overrides=cli_overrides, executor="python"
    )
    config = manifest.model_dump()
    upload_json(config, task_manifest_object_key(run.run_id, stage))
    module = _job_module(stage)
    return module.run_task(config)


def run_pipeline(
    run: PipelineRun,
    *,
    stages: list[str] | None = None,
    cli_overrides: dict[str, dict] | None = None,
    log: Callable[[str], None] = logger.info,
) -> dict[str, dict]:
    """Run stages in order, in-process Python. Returns per-stage reports."""
    names = list(stages or STAGE_ORDER)
    unknown = [s for s in names if s not in STAGE_ORDER]
    if unknown:
        known = ", ".join(STAGE_ORDER)
        raise ValueError(f"Unknown stage(s) {unknown}; expected subset of: {known}")

    results: dict[str, dict] = {}
    for name in names:
        results[name] = run_stage(name, run, cli_overrides=cli_overrides, log=log)
    return results
