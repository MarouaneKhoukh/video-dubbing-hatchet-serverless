"""TTS — batched Kokoro manifest job."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from pipeline.batch_runner import run_batched_task
from pipeline.config import settings
from pipeline.metadata import job_args_for_chunk
from pipeline.run import PipelineRun
from pipeline.runner.base import JobExecutor
from pipeline.stages._helpers import require_inputs, stage_items, timing_payload

logger = logging.getLogger(__name__)


async def run(
    run: PipelineRun,
    *,
    executor: JobExecutor | None = None,
    log: Callable[[str], None] = logger.info,
) -> dict[str, Any]:
    items = stage_items(run, "tts")
    require_inputs(items, stage="tts", upstream="translate", keys="translated_key")

    async def build_args(chunk: list[dict[str, Any]], chunk_idx: int) -> tuple[str | None, str]:
        return None, job_args_for_chunk(run.run_id, "tts", chunk_idx)

    timing = await run_batched_task(
        task_name="tts",
        cfg=settings.tts,
        items=items,
        output_key_fn=lambda i: i["dubbed_key"],
        build_args_for_chunk=build_args,
        image=settings.tts.image,
        job_name_prefix=f"tts-{run.run_id}"[:50],
        force=run.force,
        executor=executor,
        log=log,
    )
    return {"dubbed_keys": [i["dubbed_key"] for i in items], "timing": timing_payload(timing)}
