"""Sequential pipeline runner — shared by local CLI and Hatchet adapter."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from pipeline.config import get_settings

if TYPE_CHECKING:
    from pipeline.runner.base import JobExecutor

logger = logging.getLogger(__name__)

STAGE_ORDER: tuple[str, ...] = ("extract", "transcribe", "translate", "tts", "remux")


class PipelineRun(BaseModel):
    """One dubbing run: one or many input videos under the bucket /data mount."""

    video_keys: list[str]
    target_lang: str = Field(default_factory=lambda: get_settings().target_lang)
    run_id: str = "demo"
    batch_id: str = "default"
    force: bool = False


async def run_stage(
    stage: str,
    run: PipelineRun,
    *,
    executor: JobExecutor | None = None,
    cli_overrides: dict[str, dict[str, Any]] | None = None,
    log: Callable[[str], None] = logger.info,
) -> dict[str, Any]:
    """Run one pipeline stage; write task manifest before and report after."""
    from pipeline.metadata import write_task_manifest, write_task_report
    from pipeline.stages import extract, remux, transcribe, translate, tts
    from pipeline.utils import utc_now

    stage_modules = {
        "extract": extract,
        "transcribe": transcribe,
        "translate": translate,
        "tts": tts,
        "remux": remux,
    }
    module = stage_modules.get(stage)
    if module is None:
        known = ", ".join(STAGE_ORDER)
        raise ValueError(f"Unknown stage {stage!r}; expected one of: {known}")

    executor_label = getattr(executor, "platform_label", None) if executor else None
    started_at = utc_now()
    write_task_manifest(
        run,
        stage,
        cli_overrides=cli_overrides,
        executor=executor_label,
    )

    kwargs: dict[str, Any] = {"executor": executor, "log": log}

    try:
        result = await module.run(run, **kwargs)
    except Exception as exc:
        write_task_report(
            run,
            stage,
            {},
            started_at=started_at,
            failed=True,
            error=str(exc),
        )
        raise

    write_task_report(run, stage, result, started_at=started_at)
    return result


async def run_pipeline(
    run: PipelineRun,
    *,
    executor: JobExecutor | None = None,
    stages: list[str] | None = None,
    cli_overrides: dict[str, dict[str, Any]] | None = None,
    log: Callable[[str], None] = logger.info,
) -> dict[str, dict[str, Any]]:
    """Run stages in order; returns per-stage outputs."""
    names = list(stages or STAGE_ORDER)
    stage_modules = set(STAGE_ORDER)
    unknown = [s for s in names if s not in stage_modules]
    if unknown:
        known = ", ".join(STAGE_ORDER)
        raise ValueError(f"Unknown stage(s) {unknown}; expected subset of: {known}")

    results: dict[str, dict[str, Any]] = {}
    for name in names:
        log(f"=== {name} ===")
        results[name] = await run_stage(
            name,
            run,
            executor=executor,
            cli_overrides=cli_overrides,
            log=log,
        )
    return results
