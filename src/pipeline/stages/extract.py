"""Extract audio — ffmpeg shell batch on Nebius."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from pipeline.batch_runner import run_batched_task
from pipeline.config import settings
from pipeline.run import PipelineRun
from pipeline.runner.base import JobExecutor
from pipeline.stages._helpers import stage_items, timing_payload

logger = logging.getLogger(__name__)


def extract_shell(chunk: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for idx, item in enumerate(chunk):
        audio = item["audio_key"]
        parts.append(
            f"mkdir -p $(dirname '/data/{audio}') && "
            f"ffmpeg -i /data/{item['video_key']} -vn -ac 1 -ar 16000 "
            f"-y /tmp/audio_{idx}.wav && "
            f"cp /tmp/audio_{idx}.wav /data/{audio}"
        )
    return " && ".join(parts)


async def run(
    run: PipelineRun,
    *,
    executor: JobExecutor | None = None,
    log: Callable[[str], None] = logger.info,
) -> dict[str, Any]:
    items = stage_items(run, "extract")

    async def build_args(chunk: list[dict[str, Any]], chunk_idx: int) -> tuple[str | None, str]:
        return "sh", f'-c "{extract_shell(chunk)}"'

    timing = await run_batched_task(
        task_name="extract",
        cfg=settings.extract,
        items=items,
        output_key_fn=lambda i: i["audio_key"],
        build_args_for_chunk=build_args,
        image=settings.extract.image,
        job_name_prefix=f"ffmpeg-extract-{run.run_id}"[:50],
        force=run.force,
        executor=executor,
        log=log,
    )
    return {
        "audio_keys": [i["audio_key"] for i in items],
        "video_keys": [i["video_key"] for i in items],
        "stems": [i["stem"] for i in items],
        "timing": timing_payload(timing),
    }
