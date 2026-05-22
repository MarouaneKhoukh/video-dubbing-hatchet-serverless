"""Task manifests, reports, and upstream input discovery for pipeline runs."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from pipeline.config import config
from pipeline.paths import (
    RUNS_PREFIX,
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
    "extract": (),
    "transcribe": ("model", "device", "batch_size"),
    "translate": ("model", "device", "batch_size"),
    "tts": ("voice", "lang", "device", "repo", "batch_size"),
    "remux": (),
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
    video_keys: list[str] | None = None  # only populated for extract; downstream stages derive inputs from upstream reports


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


# ── Container job helpers ─────────────────────────────────────────────────────


def manifest_path_from_argv() -> Path:
    """Return the manifest path passed as the container's first argv."""
    if len(sys.argv) < 2:
        raise SystemExit("usage: <job>.py <task_manifest.json>")
    return Path(sys.argv[1].strip())


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
    task_cfg = getattr(config.pipeline, task)
    configured = getattr(task_cfg, "device", "cuda")
    return configured if task_cfg.compute.platform.startswith("gpu-") else "cpu"


def _task_config(task: str, *, cli_overrides: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    task_settings = {
        "extract": config.pipeline.extract,
        "transcribe": config.pipeline.transcribe,
        "translate": config.pipeline.translate,
        "tts": config.pipeline.tts,
        "remux": config.pipeline.remux,
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
        video_keys=list(run.video_keys) if task == "extract" else None,
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


# Output key field per task (what each stage produces).
_OUTPUT_FIELDS: dict[str, tuple[str, ...]] = {
    "extract": ("audio_key",),
    "transcribe": ("transcript_key", "aligned_key"),
    "translate": ("translated_key",),
    "tts": ("dubbed_key",),
    "remux": ("output_key",),
}


def expected_output_keys(run: PipelineRun, task: str) -> list[str]:
    """All artifact object keys this stage is expected to produce for *run*.

    Used by Hatchet pre-flight to decide whether work is needed: scan upstream
    report (or ``run.video_keys`` for extract) to derive stems, then build the
    full output-key list. Returns an empty list when upstream hasn't run.
    """
    from pipeline.paths import build_run_items, build_run_items_from_stems

    if task not in _OUTPUT_FIELDS:
        raise ValueError(f"Unknown task {task!r}")

    if task == "extract":
        items = build_run_items(list(run.video_keys), run.run_id)
    else:
        stems = resolve_upstream_stems(run.run_id, task)
        if not stems:
            return []
        items = build_run_items_from_stems(
            stems,
            run.run_id,
            video_keys_by_stem=video_keys_by_stem(run.run_id) if task == "remux" else None,
        )

    keys: list[str] = []
    for item in items:
        for field in _OUTPUT_FIELDS[task]:
            keys.append(item[field])
    return keys


def write_skipped_report(run: PipelineRun, task: str, expected: list[str]) -> dict[str, Any]:
    """Emit a status='skipped' report when Hatchet pre-flight finds nothing to do.

    Shape mirrors the report a successful ``run_task`` would write, so downstream
    pre-flight (which reads upstream report outputs) keeps working.
    """
    started = utc_now()
    field = _OUTPUT_FIELDS[task][0]
    outputs_key = {
        "extract": "audio_keys",
        "transcribe": "transcript_keys",
        "translate": "translated_keys",
        "tts": "dubbed_keys",
        "remux": "output_keys",
    }[task]
    # For transcribe we also expose aligned_keys (paired with transcript_keys).
    extra: dict[str, list[str]] = {}
    if task == "transcribe":
        n = len(expected) // 2
        # expected alternates [transcript, aligned] per stem
        extra["aligned_keys"] = expected[1::2]
        result_keys = expected[0::2]
    else:
        result_keys = expected

    result: dict[str, Any] = {outputs_key: result_keys, **extra}
    if task == "extract":
        result["video_keys"] = list(run.video_keys)
        result["stems"] = [Path(k).stem for k in result_keys]

    timing = {
        "task": task,
        "total_files": len(result_keys),
        "processed_files": 0,
        "skipped_files": len(result_keys),
        "wall_s": 0.0,
        "per_file_s": 0.0,
    }
    result["timing"] = timing
    write_task_report(run.run_id, run.batch_id, task, result, started_at=started)
    return result


def make_timing(task: str, *, total: int, processed: int, skipped: int, t0: float) -> dict[str, Any]:
    """Build the per-task timing dict embedded in a report."""
    wall_s = round(time.perf_counter() - t0, 2)
    per_file_s = round(wall_s / processed, 2) if processed else 0.0
    return {
        "task": task,
        "total_files": total,
        "processed_files": processed,
        "skipped_files": skipped,
        "wall_s": wall_s,
        "per_file_s": per_file_s,
    }


def record_task_result(
    config: dict[str, Any],
    result: dict[str, Any],
    *,
    started_at: str,
    failed: bool = False,
    error: str | None = None,
) -> None:
    """Convenience: pull run_id/batch_id/task from a manifest dict and write the report."""
    write_task_report(
        config["run_id"],
        config["batch_id"],
        config["task"],
        result,
        started_at=started_at,
        failed=failed,
        error=error,
    )


def write_task_report(
    run_id: str,
    batch_id: str,
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

    manifest_key = task_manifest_object_key(run_id, task)
    manifest = read_json(manifest_key)
    device = manifest.get("config", {}).get("device") if manifest else None

    report: dict[str, Any] = {
        "task": task,
        "run_id": run_id,
        "batch_id": batch_id,
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
    upload_json(report, task_report_object_key(run_id, task))
