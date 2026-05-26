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
import time
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


class NebiusJobError(RuntimeError):
    """Nebius job ended in a non-COMPLETED terminal state.

    Carries the full ``record`` (job_id, created_at_s, state_transitions,
    terminal_state) so callers can attach per-job cost/timeline data to their
    stage report even on the failure path.
    """

    def __init__(self, message: str, *, record: dict):
        super().__init__(message)
        self.record = record


_sdk: SDK | None = None


def _get_sdk() -> SDK:
    global _sdk
    if _sdk is None:
        token = require_cloud_setting("NEBIUS_IAM_TOKEN", secrets.nebius_iam_token)
        _sdk = SDK(credentials=token)
    return _sdk


async def _create_nebius_job(
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


async def _wait_for_job_completion(
    job_id: str,
    poll_interval: int = 15,
    max_polls: int = 120,
) -> dict:
    """Poll a Nebius job until terminal state, recording state-transition timeline.

    Returns a record on COMPLETED. Raises ``NebiusJobError`` (carrying the same
    record on ``.record``) for non-COMPLETED terminal states so callers can both
    propagate the failure to Hatchet AND extract timeline/cost data. The
    polling cadence (default 15s) bounds the resolution of the timeline.
    """
    sdk = _get_sdk()
    job_svc = JobServiceClient(sdk)

    state_transitions: list[dict] = []
    last_state = None

    for attempt in range(max_polls):
        response = await job_svc.get(GetJobRequest(id=job_id))
        state = response.status.state
        state_name = state.name

        if state != last_state:
            state_transitions.append({"state": state_name, "observed_at_s": time.time()})
            last_state = state

        logger.info(f"Job {job_id} | state: {state_name} | poll {attempt + 1}/{max_polls}")

        if state in _TERMINAL_STATES:
            record = {
                "job_id": job_id,
                "terminal_state": state_name,
                "state_transitions": state_transitions,
            }
            if state == JobStatus.State.COMPLETED:
                return record
            if state == JobStatus.State.ERROR:
                raise NebiusJobError(
                    f"Job {job_id} was preempted (state={state_name}). "
                    "Hatchet will retry on a new GPU.",
                    record=record,
                )
            raise NebiusJobError(
                f"Job {job_id} failed with state: {state_name}",
                record=record,
            )

        await asyncio.sleep(poll_interval)

    raise TimeoutError(f"Job {job_id} did not complete within the timeout period")


async def _cancel_nebius_job(
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
    """Submit a Nebius job and wait for terminal state.

    Returns a record ``{job_id, created_at_s, terminal_state, state_transitions,
    platform, preset, preemptible}`` on COMPLETED. Raises ``NebiusJobError`` on
    non-COMPLETED terminal states (the exception's ``.record`` carries the same
    data for failure-path reporting). On outer cancel / Hatchet timeout, cancels
    the cloud job and re-raises the original exception (no record attached).
    """
    created_at_s = time.time()
    job_id = await _create_nebius_job(
        name=name,
        image=image,
        args=args,
        job=job,
        container_command=container_command,
    )
    try:
        record = await _wait_for_job_completion(job_id)
    except NebiusJobError as exc:
        # Enrich the record with submission context before propagating.
        exc.record.update({
            "created_at_s": created_at_s,
            "platform": job.platform,
            "preset": job.preset,
            "preemptible": job.preemptible,
        })
        raise
    except BaseException as exc:
        # Outer cancel / Hatchet timeout / polling TimeoutError / network — cancel
        # the cloud job before propagating. asyncio.shield keeps cleanup running.
        cancel_reason = type(exc).__name__
        logger.warning(
            f"create_and_wait exiting via {cancel_reason} — cancelling Nebius job {job_id}"
        )
        try:
            await asyncio.shield(_cancel_nebius_job(job_id))
        except asyncio.CancelledError:
            # Outer cancel re-fired despite the shield. Best-effort: the shielded
            # task continues to run inside Nebius; we just stop waiting for it.
            logger.warning(
                f"Cancel-cleanup for {job_id} interrupted by re-cancellation; job state unknown"
            )
        raise

    record.update({
        "created_at_s": created_at_s,
        "platform": job.platform,
        "preset": job.preset,
        "preemptible": job.preemptible,
    })
    return record
