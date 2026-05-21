"""Remux — ffmpeg shell batch on Nebius."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from pipeline.batch_runner import run_batched_task
from pipeline.config import settings
from pipeline.run import PipelineRun
from pipeline.runner.base import JobExecutor
from pipeline.stages._helpers import require_inputs, stage_items, timing_payload

logger = logging.getLogger(__name__)


def remux_shell(chunk: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for idx, item in enumerate(chunk):
        output = item["output_key"]
        parts.append(
            f"mkdir -p $(dirname '/data/{output}') && "
            f"ffmpeg -i /data/{item['video_key']} -i /data/{item['dubbed_key']} "
            f"-map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -shortest "
            f"-y /tmp/out_{idx}.mp4 && "
            f"cp /tmp/out_{idx}.mp4 /data/{output}"
        )
    return " && ".join(parts)


async def run(
    run: PipelineRun,
    *,
    executor: JobExecutor | None = None,
    log: Callable[[str], None] = logger.info,
) -> dict[str, Any]:
    items = stage_items(run, "remux")
    require_inputs(items, stage="remux", upstream="tts", keys=("video_key", "dubbed_key"))

    async def build_args(chunk: list[dict[str, Any]], chunk_idx: int) -> tuple[str | None, str]:
        return "sh", f'-c "{remux_shell(chunk)}"'

    timing = await run_batched_task(
        task_name="remux",
        cfg=settings.remux,
        items=items,
        output_key_fn=lambda i: i["output_key"],
        build_args_for_chunk=build_args,
        image=settings.remux.image,
        job_name_prefix=f"ffmpeg-remux-{run.run_id}"[:50],
        force=run.force,
        executor=executor,
        log=log,
    )
    return {
        "output_keys": [i["output_key"] for i in items],
        "count": len(items),
        "status": "completed",
        "timing": timing_payload(timing),
    }
