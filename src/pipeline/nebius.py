"""
Nebius SDK helpers for creating and polling GPU/CPU jobs.

Key design: create_and_wait() raises RuntimeError on ERROR state.
Hatchet catches this and retries the step automatically.
This is how preemption recovery works.
"""

import asyncio
import logging
from datetime import timedelta

from nebius.sdk import SDK
from nebius.api.nebius.ai.v1 import (
    CreateJobRequest,
    GetJobRequest,
    JobServiceClient,
    JobSpec,
    JobStatus,
)
from nebius.api.nebius.common.v1 import ResourceMetadata
from nebius.api.nebius.compute.v1 import DiskSpec as ComputeDiskSpec

from pipeline.config import require_cloud_setting, settings

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
        token = require_cloud_setting("NEBIUS_IAM_TOKEN", settings.nebius_iam_token)
        _sdk = SDK(credentials=token)
    return _sdk


async def create_nebius_job(
    name: str,
    image: str,
    args: str,
    platform: str,
    preset: str,
    preemptible: bool = True,
    container_command: str | None = None,
    timeout_minutes: int | None = None,
) -> str:
    sdk = _get_sdk()
    job_svc = JobServiceClient(sdk)

    spec = JobSpec(
        image=image,
        container_command=container_command,
        args=args,
        platform=platform,
        preset=preset,
        subnet_id=require_cloud_setting("NEBIUS_SUBNET_ID", settings.nebius_subnet_id),
        timeout=timedelta(minutes=timeout_minutes or 60),
        preemptible=preemptible,
        disk=JobSpec.DiskSpec(
            type=ComputeDiskSpec.DiskType.NETWORK_SSD,
            size_bytes=settings.hardware.job_disk_gb * 1024 * 1024 * 1024,
        ),
        volumes=[
            JobSpec.VolumeMount(
                source=require_cloud_setting("NEBIUS_BUCKET_ID", settings.nebius_bucket_id),
                container_path="/data",
                mode=JobSpec.VolumeMount.Mode.READ_WRITE,
            ),
        ],
    )

    request = CreateJobRequest(
        metadata=ResourceMetadata(
            parent_id=require_cloud_setting("NEBIUS_PROJECT_ID", settings.nebius_project_id),
            name=name,
        ),
        spec=spec,
    )

    logger.info(
        f"Creating Nebius job: {name} | {platform}/{preset} | preemptible={preemptible}"
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


async def create_and_wait(
    name: str,
    image: str,
    args: str,
    platform: str,
    preset: str,
    preemptible: bool = True,
    container_command: str | None = None,
    timeout_minutes: int | None = None,
) -> dict:
    job_id = await create_nebius_job(
        name=name,
        image=image,
        args=args,
        platform=platform,
        preset=preset,
        preemptible=preemptible,
        container_command=container_command,
        timeout_minutes=timeout_minutes,
    )
    return await wait_for_job_completion(job_id)
