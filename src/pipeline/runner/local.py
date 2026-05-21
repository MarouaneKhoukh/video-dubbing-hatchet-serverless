"""Local Docker executor — same /data layout as Nebius bucket mount."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from pipeline.config import ComputeConfig

logger = logging.getLogger(__name__)

# Map cloud image tags to locally built tags (see DEVELOPER_GUIDE Level 1b).
DEFAULT_LOCAL_IMAGES: dict[str, str] = {
    "mnrozhkov/nebius-transcribe:v0.1.0": "video-dubbing-transcribe:local",
    "mnrozhkov/nebius-translate:v0.1.0": "video-dubbing-translate:local",
    "mnrozhkov/nebius-tts:v0.1.0": "video-dubbing-tts:local",
}


class LocalExecutor:
    """Run pipeline chunks via ``docker run`` with ``./data`` mounted at ``/data``."""

    platform_label = "local-docker"

    def __init__(
        self,
        data_dir: Path,
        *,
        models_dir: Path | None = None,
        use_gpu: bool = False,
        image_map: dict[str, str] | None = None,
    ) -> None:
        self.data_dir = data_dir.resolve()
        self.models_dir = models_dir.resolve() if models_dir else self.data_dir / "models"
        self.use_gpu = use_gpu
        self.image_map = {**DEFAULT_LOCAL_IMAGES, **(image_map or {})}

    def resolve_image(self, image: str) -> str:
        return self.image_map.get(image, image)

    @staticmethod
    def _sh_script_args(args: str) -> list[str]:
        """Parse Nebius ``sh`` args (``-c "script"``) into docker argv tail."""
        script = args
        if script.startswith("-c "):
            script = script[3:].strip()
            if len(script) >= 2 and script[0] == script[-1] and script[0] in "\"'":
                script = script[1:-1]
        return ["-c", script]

    def _docker_cmd(
        self,
        image: str,
        container_command: str | None,
        args: str,
        compute: ComputeConfig,
    ) -> list[str]:
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{self.data_dir}:/data",
            "-v",
            f"{self.models_dir}:/data/models",
        ]
        if self.use_gpu and compute.gpu:
            cmd.extend(["--gpus", "all"])

        if container_command:
            cmd.extend(["--entrypoint", container_command])

        cmd.append(image)

        if container_command == "sh":
            cmd.extend(self._sh_script_args(args))
        elif " " in args.strip():
            cmd.extend(args.split())
        else:
            cmd.append(args)

        return cmd

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
        resolved = self.resolve_image(image)
        cmd = self._docker_cmd(resolved, container_command, args, compute)
        logger.info(f"[local] {name} | {resolved} | {' '.join(cmd[2:])}")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout_minutes * 60,
            )
        except TimeoutError as exc:
            proc.kill()
            await proc.wait()
            raise TimeoutError(
                f"Local container {name!r} exceeded {timeout_minutes}m timeout"
            ) from exc

        if stdout:
            logger.info(stdout.decode(errors="replace").rstrip())
        if proc.returncode != 0:
            err = stderr.decode(errors="replace").rstrip()
            raise RuntimeError(
                f"Local container {name!r} failed (exit {proc.returncode}): {err}"
            )

        return {"job_id": name, "state": "COMPLETED"}
