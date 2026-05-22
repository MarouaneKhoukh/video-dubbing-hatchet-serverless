"""
Hatchet workflow — one task per pipeline stage; each launches one Nebius job.

Per-task flow:
    1. Pre-flight: scan S3 for expected outputs. If all present (and not forced),
       skip Nebius launch and write a status="skipped" report.
    2. Otherwise, write the task manifest and call ``create_and_wait`` to run
       one Nebius serverless job with the manifest path as its argv.
    3. Post-flight: re-scan S3; raise if any expected output is still missing.

Preemption recovery is automatic: ``create_and_wait`` raises ``RuntimeError`` on
ERROR state → Hatchet retries → pre-flight on the retry sees the partial S3
state → launches a new Nebius job that skips already-completed files (per-file
``force=False`` idempotency inside ``run_task``).
"""

from __future__ import annotations

from hatchet_sdk import Context, Hatchet

from pipeline.config import StageConfig, config
from pipeline.metadata import (
    expected_output_keys,
    write_skipped_report,
    write_task_manifest,
)
from pipeline.nebius import create_and_wait
from pipeline.paths import task_manifest_container_path, task_report_object_key
from pipeline.run import PipelineRun
from pipeline.storage import object_exists


def _hatchet_timeout(stage: StageConfig) -> str:
    return f"{stage.compute.job_timeout_min * 60 + config.timeout_buffer_s}s"


hatchet = Hatchet(debug=True)

workflow = hatchet.workflow(
    name=config.workflow_name,
    on_events=["dubbing:batch"],
    input_validator=PipelineRun,
)


async def _run_remote(task: str, run: PipelineRun, ctx: Context) -> dict:
    cfg = getattr(config.pipeline, task)

    expected = expected_output_keys(run, task)
    missing = [k for k in expected if not object_exists(k)]

    if expected and not missing and not run.force:
        ctx.log(f"[{task}] all {len(expected)} outputs present; skipping Nebius launch")
        write_skipped_report(run, task, expected)
        return {
            "task": task,
            "report_key": task_report_object_key(run.run_id, task),
            "status": "skipped",
            "processed": 0,
            "skipped": len(expected),
        }

    ctx.log(
        f"[{task}] {len(missing) or len(expected)}/{len(expected) or '?'} outputs missing "
        f"→ launching Nebius job ({cfg.compute.platform}/{cfg.compute.preset})"
    )
    write_task_manifest(run, task, executor="nebius")

    attempt = getattr(ctx, "retry_count", 0)
    await create_and_wait(
        name=f"{task}-{run.run_id}-{attempt}"[:50],
        image=f"{cfg.image_name}:{config.pipeline.image_tag}",
        args=task_manifest_container_path(run.run_id, task),
        job=cfg.compute,
    )

    # Re-compute expected (extract populates it; downstream needs upstream report).
    expected = expected_output_keys(run, task)
    still_missing = [k for k in expected if not object_exists(k)]
    if still_missing:
        sample = ", ".join(still_missing[:5]) + ("…" if len(still_missing) > 5 else "")
        raise RuntimeError(
            f"[{task}] job completed but {len(still_missing)}/{len(expected)} outputs missing: {sample}"
        )

    return {
        "task": task,
        "report_key": task_report_object_key(run.run_id, task),
        "status": "completed",
        "processed": len(missing) if missing else len(expected),
        "skipped": len(expected) - (len(missing) if missing else len(expected)),
    }


@workflow.task(
    execution_timeout=_hatchet_timeout(config.pipeline.extract),
    retries=config.stages.extract.retries,
)
async def extract(run: PipelineRun, ctx: Context) -> dict:
    return await _run_remote("extract", run, ctx)


@workflow.task(
    parents=[extract],
    execution_timeout=_hatchet_timeout(config.pipeline.transcribe),
    retries=config.stages.transcribe.retries,
)
async def transcribe(run: PipelineRun, ctx: Context) -> dict:
    return await _run_remote("transcribe", run, ctx)


@workflow.task(
    parents=[transcribe],
    execution_timeout=_hatchet_timeout(config.pipeline.translate),
    retries=config.stages.translate.retries,
)
async def translate(run: PipelineRun, ctx: Context) -> dict:
    return await _run_remote("translate", run, ctx)


@workflow.task(
    parents=[translate],
    execution_timeout=_hatchet_timeout(config.pipeline.tts),
    retries=config.stages.tts.retries,
)
async def tts(run: PipelineRun, ctx: Context) -> dict:
    return await _run_remote("tts", run, ctx)


@workflow.task(
    parents=[tts],
    execution_timeout=_hatchet_timeout(config.pipeline.remux),
    retries=config.stages.remux.retries,
)
async def remux(run: PipelineRun, ctx: Context) -> dict:
    return await _run_remote("remux", run, ctx)
