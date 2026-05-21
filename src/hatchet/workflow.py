"""
Hatchet workflow — thin orchestration over pipeline stages.

Each task delegates to ``pipeline.run.run_stage``; batching, manifests,
and Nebius fan-out live in the pipeline layer.
"""

from __future__ import annotations

from hatchet_sdk import (
    ConcurrencyExpression,
    ConcurrencyLimitStrategy,
    Context,
    Hatchet,
)

from pipeline.config import settings
from pipeline.run import PipelineRun, run_stage

hatchet = Hatchet(debug=True)

workflow = hatchet.workflow(
    name="video-dubbing-batch-pipeline",
    on_events=["dubbing:batch"],
    input_validator=PipelineRun,
    concurrency=ConcurrencyExpression(
        expression="input.batch_id",
        max_runs=settings.max_concurrent_batches,
        limit_strategy=ConcurrencyLimitStrategy.GROUP_ROUND_ROBIN,
    ),
)


@workflow.task(
    execution_timeout=f"{settings.extract.hatchet_timeout_s}s",
    retries=settings.extract.retries,
)
async def extract_audio(run: PipelineRun, ctx: Context) -> dict:
    return await run_stage("extract", run, log=ctx.log)


@workflow.task(
    parents=[extract_audio],
    execution_timeout=f"{settings.transcribe.hatchet_timeout_s}s",
    retries=settings.transcribe.retries,
)
async def transcribe(run: PipelineRun, ctx: Context) -> dict:
    return await run_stage("transcribe", run, log=ctx.log)


@workflow.task(
    parents=[transcribe],
    execution_timeout=f"{settings.translate.hatchet_timeout_s}s",
    retries=settings.translate.retries,
)
async def translate_text(run: PipelineRun, ctx: Context) -> dict:
    return await run_stage("translate", run, log=ctx.log)


@workflow.task(
    parents=[translate_text],
    execution_timeout=f"{settings.tts.hatchet_timeout_s}s",
    retries=settings.tts.retries,
)
async def synthesize_tts(run: PipelineRun, ctx: Context) -> dict:
    return await run_stage("tts", run, log=ctx.log)


@workflow.task(
    parents=[synthesize_tts],
    execution_timeout=f"{settings.remux.hatchet_timeout_s}s",
    retries=settings.remux.retries,
)
async def remux_video(run: PipelineRun, ctx: Context) -> dict:
    return await run_stage("remux", run, log=ctx.log)
