"""Executor protocol — run one batched chunk as a container job."""

from __future__ import annotations

from typing import Protocol

from pipeline.config import ComputeConfig


class JobExecutor(Protocol):
    """Run a single pipeline chunk (ffmpeg shell or manifest job)."""

    platform_label: str

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
        """Execute the chunk; raise on failure."""
        ...
