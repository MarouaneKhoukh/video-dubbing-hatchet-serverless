"""Nebius serverless job executor."""

from __future__ import annotations

from pipeline.config import ComputeConfig
from pipeline.nebius import create_and_wait


class NebiusExecutor:
    platform_label = "nebius"

    async def run_chunk(
        self,
        *,
        name: str,
        image: str,
        container_command: str | None,
        args: str,
        compute: ComputeConfig,
        timeout_minutes: int,
    ) -> dict:
        return await create_and_wait(
            name=name,
            image=image,
            container_command=container_command,
            args=args,
            platform=compute.platform,
            preset=compute.preset,
            preemptible=compute.preemptible,
            timeout_minutes=timeout_minutes,
        )
