"""
Per-task batch runner — filter, chunk, fan-out, and record execution timing.

Timing logs are structured for platform/batch-size experiments: each Hatchet
task logs wall time per Nebius chunk and a summary line with platform, preset,
batch_size, and file counts. Summary is also returned for Hatchet task output.
"""

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Iterable

from pipeline.config import TaskConfig
from pipeline.runner.base import JobExecutor
from pipeline.runner.nebius import NebiusExecutor
from pipeline.storage import item_outputs_exist, output_keys, unprocessed_items

logger = logging.getLogger(__name__)


BuildArgsForChunk = Callable[
    [list[dict[str, Any]], int],
    Awaitable[tuple[str | None, str]],
]


@dataclass
class ChunkTiming:
    chunk_idx: int
    file_count: int
    wall_s: float
    job_id: str


@dataclass
class TaskTiming:
    task: str
    platform: str
    preset: str
    gpu: bool
    preemptible: bool
    batch_size: int
    max_concurrent: int
    total_files: int
    processed_files: int
    skipped_files: int
    chunk_count: int
    wall_s: float
    chunks: list[ChunkTiming] = field(default_factory=list)

    def summary_line(self) -> str:
        chunk_times = [f"{c.wall_s:.1f}s" for c in self.chunks]
        per_file = (
            self.wall_s / self.processed_files if self.processed_files else 0.0
        )
        return (
            f"[timing] task={self.task} "
            f"platform={self.platform} preset={self.preset} "
            f"gpu={self.gpu} preemptible={self.preemptible} "
            f"batch_size={self.batch_size} max_concurrent={self.max_concurrent} "
            f"files={self.processed_files}/{self.total_files} skipped={self.skipped_files} "
            f"chunks={self.chunk_count} wall_s={self.wall_s:.1f} "
            f"per_file_s={per_file:.1f} chunk_times=[{', '.join(chunk_times)}]"
        )


def _chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


async def run_batched_task(
    *,
    task_name: str,
    cfg: TaskConfig,
    items: list[dict[str, Any]],
    output_key_fn: Callable[[dict[str, Any]], str | Iterable[str]],
    build_args_for_chunk: BuildArgsForChunk,
    image: str,
    job_name_prefix: str,
    force: bool = False,
    executor: JobExecutor | None = None,
    log: Callable[[str], None] = logger.info,
) -> TaskTiming:
    """Filter, chunk, fan-out. Returns timing stats; raises if outputs missing."""
    compute = cfg.compute
    job_executor = executor or NebiusExecutor()
    platform = getattr(job_executor, "platform_label", compute.platform)
    t0 = time.perf_counter()

    timing = TaskTiming(
        task=task_name,
        platform=platform,
        preset=compute.preset,
        gpu=compute.gpu,
        preemptible=compute.preemptible,
        batch_size=cfg.batch_size,
        max_concurrent=cfg.max_concurrent,
        total_files=len(items),
        processed_files=0,
        skipped_files=0,
        chunk_count=0,
        wall_s=0.0,
    )

    if not items:
        log(f"[{task_name}] no input items; nothing to do.")
        timing.wall_s = time.perf_counter() - t0
        log(timing.summary_line())
        return timing

    if force:
        todo = list(items)
        timing.skipped_files = 0
        timing.processed_files = len(todo)
        log(f"[{task_name}] force=True — reprocessing all {len(todo)} item(s)")
    else:
        todo = unprocessed_items(items, output_key_fn)
        timing.skipped_files = timing.total_files - len(todo)
        timing.processed_files = len(todo)

        log(
            f"[{task_name}] {timing.total_files} items | "
            f"{timing.skipped_files} already done | {timing.processed_files} to process | "
            f"compute={compute.platform}/{compute.preset} gpu={compute.gpu} "
            f"batch_size={cfg.batch_size}"
        )

    if not todo:
        timing.wall_s = time.perf_counter() - t0
        log(timing.summary_line())
        return timing

    all_chunks = _chunked(items, cfg.batch_size)

    def _chunk_needs_work(chunk: list[dict[str, Any]]) -> bool:
        if force:
            return True
        return any(not item_outputs_exist(item, output_key_fn) for item in chunk)

    chunks_to_run = [(idx, chunk) for idx, chunk in enumerate(all_chunks) if _chunk_needs_work(chunk)]
    timing.chunk_count = len(chunks_to_run)
    log(
        f"[{task_name}] launching {len(chunks_to_run)} job(s) "
        f"(max_concurrent={cfg.max_concurrent}, platform={platform})"
    )

    sem = asyncio.Semaphore(cfg.max_concurrent)

    async def _run_one(chunk_idx: int, chunk: list[dict[str, Any]]) -> ChunkTiming:
        async with sem:
            chunk_t0 = time.perf_counter()
            container_command, args = await build_args_for_chunk(chunk, chunk_idx)
            log(
                f"[{task_name}] chunk {chunk_idx:03d} | "
                f"{len(chunk)} files in slice | launching"
            )
            result = await job_executor.run_chunk(
                name=f"{job_name_prefix}-{chunk_idx:03d}",
                image=image,
                container_command=container_command,
                args=args,
                compute=compute,
                timeout_minutes=cfg.job_timeout_min,
            )
            wall_s = time.perf_counter() - chunk_t0
            chunk_timing = ChunkTiming(
                chunk_idx=chunk_idx,
                file_count=len(chunk),
                wall_s=wall_s,
                job_id=result.get("job_id", ""),
            )
            log(
                f"[{task_name}] chunk {chunk_idx:03d} done | "
                f"wall_s={wall_s:.1f} | {len(chunk)} files | "
                f"per_file_s={wall_s / len(chunk):.1f}"
            )
            return chunk_timing

    timing.chunks = list(
        await asyncio.gather(*[_run_one(idx, chunk) for idx, chunk in chunks_to_run])
    )

    missing_items = [item for item in todo if not item_outputs_exist(item, output_key_fn)]
    if missing_items:
        missing_keys = [
            key for item in missing_items for key in output_keys(output_key_fn, item)
        ]
        sample = ", ".join(missing_keys[:5]) + ("…" if len(missing_keys) > 5 else "")
        raise RuntimeError(
            f"[{task_name}] batch completed but {len(missing_items)}/{len(todo)} "
            f"items still missing outputs: {sample}"
        )

    timing.wall_s = time.perf_counter() - t0
    log(f"[{task_name}] all {len(todo)} outputs verified")
    log(timing.summary_line())
    logger.info(json.dumps({"timing": asdict(timing)}, default=str))
    return timing
