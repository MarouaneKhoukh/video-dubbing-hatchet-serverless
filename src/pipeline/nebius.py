"""
Nebius SDK helpers for creating, polling, and cancelling GPU/CPU jobs.

Key design:
  - ``create_and_wait()`` raises ``RuntimeError`` on ERROR state. Hatchet catches
    this and retries the step automatically — preemption recovery flow.
  - If the awaiting task is cancelled (e.g. Hatchet ``execution_timeout`` fires,
    worker shutdown, or our polling ``TimeoutError``), ``create_and_wait()``
    cancels the Nebius job and waits for it to reach a terminal state before
    re-raising. No orphaned jobs.
"""

import asyncio
import logging
from datetime import timedelta

from nebius.sdk import SDK
from nebius.api.nebius.ai.v1 import (
    CancelJobRequest,
    CreateJobRequest,
    GetJobRequest,
    JobServiceClient,
    JobSpec,
    JobStatus,
)
from nebius.api.nebius.common.v1 import ResourceMetadata
from nebius.api.nebius.compute.v1 import DiskSpec as ComputeDiskSpec

from pipeline.config import Compute, require_cloud_setting, secrets

logger = logging.getLogger(__name__)

_TERMINAL_STATES = {
    JobStatus.State.COMPLETED,
    JobStatus.State.FAILED,
    JobStatus.State.CANCELLED,
    JobStatus.State.ERROR,
}


_sdk: SDK | None = None


def _get_sdk() -> SDK:
    global _sdk
    if _sdk is None:
        token = require_cloud_setting("NEBIUS_IAM_TOKEN", secrets.nebius_iam_token)
        _sdk = SDK(credentials=token)
    return _sdk


async def create_nebius_job(
    name: str,
    image: str,
    args: str,
    job: Compute,
    container_command: str | None = None,
) -> str:
    sdk = _get_sdk()
    job_svc = JobServiceClient(sdk)

    spec = JobSpec(
        image=image,
        container_command=container_command,
        args=args,
        platform=job.platform,
        preset=job.preset,
        subnet_id=require_cloud_setting("NEBIUS_SUBNET_ID", secrets.nebius_subnet_id),
        timeout=timedelta(minutes=job.job_timeout_min),
        preemptible=job.preemptible,
        disk=JobSpec.DiskSpec(
            type=ComputeDiskSpec.DiskType.NETWORK_SSD,
            size_bytes=job.job_disk_gb * 1024 * 1024 * 1024,
        ),
        volumes=[
            JobSpec.VolumeMount(
                source=require_cloud_setting("NEBIUS_BUCKET_ID", secrets.nebius_bucket_id),
                container_path="/data",
                mode=JobSpec.VolumeMount.Mode.READ_WRITE,
            ),
        ],
    )

    request = CreateJobRequest(
        metadata=ResourceMetadata(
            parent_id=require_cloud_setting("NEBIUS_PROJECT_ID", secrets.nebius_project_id),
            name=name,
        ),
        spec=spec,
    )

    logger.info(
        f"Creating Nebius job: {name} | {job.platform}/{job.preset} | preemptible={job.preemptible}"
    )
    operation = await job_svc.create(request)
    await operation.wait()
    job_id = operation.resource_id
    logger.info(f"Job created: {job_id}")
    return job_id


async def wait_for_job_completion(
    job_id: str,
    poll_interval: int = 15,
    max_polls: int = 120,
) -> dict:
    sdk = _get_sdk()
    job_svc = JobServiceClient(sdk)

    for attempt in range(max_polls):
        response = await job_svc.get(GetJobRequest(id=job_id))
        state = response.status.state
        state_name = state.name

        logger.info(f"Job {job_id} | state: {state_name} | poll {attempt + 1}/{max_polls}")

        if state in _TERMINAL_STATES:
            if state == JobStatus.State.COMPLETED:
                return {"job_id": job_id, "state": state_name}

            if state == JobStatus.State.ERROR:
                # Preemption case - raise so Hatchet retries the step
                raise RuntimeError(
                    f"Job {job_id} was preempted (state={state_name}). "
                    "Hatchet will retry on a new GPU."
                )

            raise RuntimeError(
                f"Job {job_id} failed with state: {state_name}"
            )

        await asyncio.sleep(poll_interval)

    raise TimeoutError(f"Job {job_id} did not complete within the timeout period")


async def cancel_nebius_job(
    job_id: str,
    *,
    poll_interval: int = 5,
    max_polls: int = 60,
) -> None:
    """Send a cancel request and poll until the job reaches a terminal state.

    Tolerant: a job that's already terminal still returns cleanly; transient SDK
    errors are logged but don't propagate (caller is typically in a failure path).
    """
    sdk = _get_sdk()
    job_svc = JobServiceClient(sdk)

    try:
        logger.info(f"Cancelling Nebius job {job_id}")
        operation = await job_svc.cancel(CancelJobRequest(id=job_id))
        await operation.wait()
    except Exception as exc:
        logger.warning(f"Cancel request for {job_id} did not succeed: {exc}")

    for attempt in range(max_polls):
        try:
            response = await job_svc.get(GetJobRequest(id=job_id))
            state = response.status.state
            logger.info(
                f"Job {job_id} | post-cancel state: {state.name} | poll {attempt + 1}/{max_polls}"
            )
            if state in _TERMINAL_STATES:
                return
        except Exception as exc:
            logger.warning(f"Polling cancelled job {job_id} failed: {exc}")
            return
        await asyncio.sleep(poll_interval)

    logger.warning(
        f"Job {job_id} did not reach a terminal state within {poll_interval * max_polls}s after cancel"
    )


async def create_and_wait(
    name: str,
    image: str,
    args: str,
    job: Compute,
    container_command: str | None = None,
) -> dict:
    job_id = await create_nebius_job(
        name=name,
        image=image,
        args=args,
        job=job,
        container_command=container_command,
    )
    try:
        return await wait_for_job_completion(job_id)
    except BaseException as exc:
        # Anything that exits the wait abnormally — Hatchet timeout / outer task
        # cancel (CancelledError), our polling TimeoutError, Nebius ERROR/FAILED
        # raise, network errors — should cancel the cloud job before propagating.
        # asyncio.shield keeps the cleanup running even when we're being cancelled.
        cancel_reason = type(exc).__name__
        logger.warning(
            f"create_and_wait exiting via {cancel_reason} — cancelling Nebius job {job_id}"
        )
        try:
            await asyncio.shield(cancel_nebius_job(job_id))
        except asyncio.CancelledError:
            # Outer cancel re-fired despite the shield. Best-effort: the shielded
            # task continues to run inside Nebius; we just stop waiting for it.
            logger.warning(
                f"Cancel-cleanup for {job_id} interrupted by re-cancellation; job state unknown"
            )
        raise
