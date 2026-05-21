"""Task manifests, reports, and upstream input discovery for pipeline runs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from pipeline.config import settings
from pipeline.paths import (
    RUNS_PREFIX,
    build_run_items_from_stems,
    task_manifest_container_path,
    task_manifest_object_key,
    task_report_object_key,
)
from pipeline.storage import list_objects, read_json, upload_json
from pipeline.utils import utc_now

if TYPE_CHECKING:
    from pipeline.run import PipelineRun

STAGE_TASKS: tuple[str, ...] = ("extract", "transcribe", "translate", "tts", "remux")

StageStatus = Literal["completed", "failed", "skipped"]

_MANIFEST_TOP_LEVEL_KEYS = ("task", "run_id", "config", "force", "target_lang")

_TASK_REQUIRED_CONFIG: dict[str, tuple[str, ...]] = {
    "transcribe": ("model", "device", "batch_size"),
    "translate": ("model", "device", "batch_size"),
    "tts": ("voice", "lang", "device", "repo", "batch_size"),
}

# Upstream task report field → artifact directory to scan as fallback.
_UPSTREAM: dict[str, tuple[str, str, str]] = {
    "transcribe": ("extract", "audio_keys", "extract"),
    "translate": ("transcribe", "transcript_keys", "transcribe"),
    "tts": ("translate", "translated_keys", "translate"),
    "remux": ("tts", "dubbed_keys", "tts"),
}


class TaskManifest(BaseModel):
    """Config snapshot written before a stage runs."""

    task: str
    run_id: str
    batch_id: str
    input_count: int
    target_lang: str
    force: bool = False
    created_at: str
    executor: str | None = None
    config: dict[str, Any]


def read_task_report(run_id: str, task: str) -> dict[str, Any] | None:
    return read_json(task_report_object_key(run_id, task))


def _stems_from_report(run_id: str, upstream_task: str, output_key: str) -> list[str] | None:
    report = read_task_report(run_id, upstream_task)
    if report is None:
        return None
    keys = report.get("outputs", {}).get(output_key)
    if not keys:
        return None
    return sorted(Path(str(key)).stem for key in keys)


def _stems_from_artifacts(run_id: str, artifact_task: str) -> list[str]:
    prefix = f"{RUNS_PREFIX}/{run_id}/{artifact_task}/"
    keys = list_objects(prefix)
    if artifact_task == "extract":
        return sorted(Path(k).stem for k in keys if k.endswith(".wav"))
    if artifact_task == "transcribe":
        return sorted(
            Path(k).stem
            for k in keys
            if k.endswith(".txt") and not k.endswith("_aligned.json")
        )
    if artifact_task == "translate":
        return sorted(Path(k).stem for k in keys if k.endswith(".txt"))
    if artifact_task == "tts":
        return sorted(Path(k).stem for k in keys if k.endswith(".wav"))
    return []


def resolve_upstream_stems(run_id: str, stage: str) -> list[str]:
    """Return ordered stems for a downstream stage, or ``[]`` if none found."""
    if stage == "extract":
        return []
    if stage not in _UPSTREAM:
        raise ValueError(f"resolve_upstream_stems does not support stage {stage!r}")

    upstream_task, output_key, artifact_task = _UPSTREAM[stage]
    stems = _stems_from_report(run_id, upstream_task, output_key)
    if stems is None:
        stems = _stems_from_artifacts(run_id, artifact_task)
    return stems


def resolve_manifest_stems(manifest: dict[str, Any]) -> list[str]:
    """Like ``resolve_upstream_stems`` but raises when inputs are missing."""
    task = manifest["task"]
    run_id = manifest["run_id"]
    stems = resolve_upstream_stems(run_id, task)
    if stems:
        return stems
    upstream_task, _, _ = _UPSTREAM[task]
    raise FileNotFoundError(
        f"[{task}] no inputs found for run {run_id!r}; "
        f"run `{upstream_task}` first or check runs/{run_id}/reports/{upstream_task}.json"
    )


# ── Container batch job helpers ───────────────────────────────────────────────


def parse_manifest_argv() -> tuple[Path, int]:
    """Parse ``<manifest.json> [<chunk_idx>]`` or a single combined arg from Nebius."""
    if len(sys.argv) < 2:
        raise SystemExit("usage: <job>.py <task_manifest.json> [chunk_idx]")
    if len(sys.argv) >= 3:
        return Path(sys.argv[1]), int(sys.argv[2])
    combined = sys.argv[1].strip()
    if " " in combined:
        path_str, idx_str = combined.split(maxsplit=1)
        return Path(path_str), int(idx_str)
    return Path(combined), 0


def _manifest_object_key(path: Path) -> str:
    key = path.as_posix()
    if key.startswith("/data/"):
        return key[len("/data/") :]
    return key.lstrip("/")


def load_manifest(path: Path) -> dict[str, Any]:
    """Load a task manifest JSON from a container or host path under ``/data``."""
    data = read_json(_manifest_object_key(path))
    if data is None:
        raise FileNotFoundError(f"manifest not found: {path}")
    return data


def chunk_items(items: list[dict[str, str]], batch_size: int, chunk_idx: int) -> list[dict[str, str]]:
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    start = chunk_idx * batch_size
    if start >= len(items):
        raise IndexError(f"chunk_idx {chunk_idx} out of range for {len(items)} items")
    return items[start : start + batch_size]


def resolve_chunk(manifest: dict[str, Any], chunk_idx: int) -> list[dict[str, str]]:
    """Return per-file item dicts for one manifest batch chunk."""
    runtime = parse_task_runtime(manifest, manifest["task"])
    stems = resolve_manifest_stems(manifest)
    items = build_run_items_from_stems(stems, runtime["run_id"])
    batch_size = int(runtime["config"]["batch_size"])
    return chunk_items(items, batch_size, chunk_idx)


def parse_task_runtime(manifest: dict[str, Any], task: str) -> dict[str, Any]:
    """Validate manifest fields required at container runtime."""
    missing_top = [key for key in _MANIFEST_TOP_LEVEL_KEYS if key not in manifest]
    if missing_top:
        raise KeyError(f"manifest missing required fields: {missing_top}")

    if manifest["task"] != task:
        raise ValueError(
            f"manifest task {manifest['task']!r} does not match expected {task!r}"
        )

    cfg = manifest["config"]
    if not isinstance(cfg, dict):
        raise TypeError("manifest config must be a dict")

    required = _TASK_REQUIRED_CONFIG.get(task)
    if required is None:
        raise ValueError(f"parse_task_runtime does not support task {task!r}")

    missing_cfg = [key for key in required if key not in cfg]
    if missing_cfg:
        raise KeyError(f"manifest config missing required fields: {missing_cfg}")

    return {
        "task": manifest["task"],
        "run_id": manifest["run_id"],
        "target_lang": manifest["target_lang"],
        "force": bool(manifest["force"]),
        "config": cfg,
    }


def config_str(cfg: dict[str, Any], key: str) -> str:
    value = cfg[key]
    if not isinstance(value, str):
        raise TypeError(f"manifest config[{key!r}] must be a string, got {type(value).__name__}")
    return value


def ensure_torch_device(device: str) -> str:
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported manifest config.device {device!r}; expected 'cpu' or 'cuda'")
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("manifest config.device='cuda' but CUDA is not available in container")
    return device


# ── Orchestrator manifest / report I/O ───────────────────────────────────────


def video_keys_by_stem(run_id: str) -> dict[str, str]:
    """Map stem → source ``video_key`` from the extract task report, when available."""
    report = read_task_report(run_id, "extract")
    if report is None:
        return {}
    outputs = report.get("outputs", {})
    stems = outputs.get("stems")
    video_keys = outputs.get("video_keys")
    if not stems or not video_keys or len(stems) != len(video_keys):
        return {}
    return dict(zip(stems, video_keys, strict=True))


def upstream_stage_name(stage: str) -> str | None:
    if stage == "extract":
        return None
    entry = _UPSTREAM.get(stage)
    return entry[0] if entry else None


def resolve_task_device(task: str) -> str:
    """Runtime device for a task manifest (``cpu`` / ``cuda``)."""
    if task in ("extract", "remux"):
        return "cpu"
    task_cfg = getattr(settings, task)
    configured = getattr(task_cfg, "device", "cuda")
    return configured if task_cfg.compute.gpu else "cpu"


def _task_config(task: str, *, cli_overrides: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    task_settings = {
        "extract": settings.extract,
        "transcribe": settings.transcribe,
        "translate": settings.translate,
        "tts": settings.tts,
        "remux": settings.remux,
    }
    if task not in task_settings:
        raise ValueError(f"Unknown task {task!r}")

    cfg = task_settings[task].model_dump()
    overrides = (cli_overrides or {}).get(task, {})
    cfg.update({key: value for key, value in overrides.items() if key != "device"})
    cfg["device"] = overrides["device"] if "device" in overrides else resolve_task_device(task)
    return cfg


def _input_count(run: PipelineRun, task: str) -> int:
    if task == "extract":
        return len(run.video_keys)
    stems = resolve_upstream_stems(run.run_id, task)
    return len(stems) if stems else len(run.video_keys)


def build_task_manifest(
    run: PipelineRun,
    task: str,
    *,
    cli_overrides: dict[str, dict[str, Any]] | None = None,
    executor: str | None = None,
) -> TaskManifest:
    return TaskManifest(
        task=task,
        run_id=run.run_id,
        batch_id=run.batch_id,
        input_count=_input_count(run, task),
        target_lang=run.target_lang,
        force=run.force,
        created_at=utc_now(),
        executor=executor,
        config=_task_config(task, cli_overrides=cli_overrides),
    )


def write_task_manifest(
    run: PipelineRun,
    task: str,
    *,
    cli_overrides: dict[str, dict[str, Any]] | None = None,
    executor: str | None = None,
) -> str:
    """Write task manifest JSON; return container path under ``/data``."""
    manifest = build_task_manifest(
        run,
        task,
        cli_overrides=cli_overrides,
        executor=executor,
    )
    upload_json(manifest.model_dump(), task_manifest_object_key(run.run_id, task))
    return task_manifest_container_path(run.run_id, task)


def job_args_for_chunk(run_id: str, task: str, chunk_idx: int) -> str:
    """Container argv tail: task manifest path + chunk index."""
    return f"{task_manifest_container_path(run_id, task)} {chunk_idx}"


def _stage_status(result: dict[str, Any], *, failed: bool) -> StageStatus:
    if failed:
        return "failed"
    timing = result.get("timing")
    if not timing:
        return "completed"
    if (
        timing.get("processed_files", 0) == 0
        and timing.get("skipped_files", 0) == timing.get("total_files", 0)
        and timing.get("total_files", 0) > 0
    ):
        return "skipped"
    return "completed"


def write_task_report(
    run: PipelineRun,
    task: str,
    result: dict[str, Any],
    *,
    started_at: str,
    failed: bool = False,
    error: str | None = None,
) -> None:
    """Write one task report JSON (overwrites prior report for this task)."""
    timing = result.get("timing")
    outputs = {key: value for key, value in result.items() if key != "timing"}
    status = _stage_status(result, failed=failed)

    manifest_key = task_manifest_object_key(run.run_id, task)
    manifest = read_json(manifest_key)
    device = manifest.get("config", {}).get("device") if manifest else None

    report: dict[str, Any] = {
        "task": task,
        "run_id": run.run_id,
        "batch_id": run.batch_id,
        "status": status,
        "device": device,
        "started_at": started_at,
        "completed_at": utc_now(),
        "manifest_key": manifest_key,
        "wall_s": timing.get("wall_s") if timing else None,
        "timing": timing,
        "outputs": outputs,
        "error": error,
    }
    upload_json(report, task_report_object_key(run.run_id, task))
