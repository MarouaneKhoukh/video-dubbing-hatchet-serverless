#!/usr/bin/env python3
"""
Pre-download pipeline models into the shared model cache.

Host (Typer + Rich):
    python scripts/download_models.py all
    python scripts/download_models.py status

Docker (plain argparse — typer/rich not installed in task images):
    docker run --rm -v $(pwd)/data/models:/data/models --entrypoint python3 \\
      video-dubbing-transcribe:local /download_models.py transcribe
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

TaskName = Literal["transcribe", "translate", "tts"]


class StepOutcome(str, Enum):
    CACHED = "cached"
    DOWNLOADED = "downloaded"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True)
class ModelRow:
    task: TaskName
    label: str
    cached: bool


@dataclass
class StepResult:
    task: TaskName
    outcome: StepOutcome
    model: str
    detail: str = ""


def _default_host_cache() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "models"


def _import_cache():
    try:
        import model_cache
    except ImportError:
        if str(Path(__file__).resolve().parent.parent) not in sys.path:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from models import model_cache
    return model_cache


def ensure_cache_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        os.environ["MODEL_CACHE_DIR"] = str(explicit)
    elif not os.environ.get("MODEL_CACHE_DIR"):
        os.environ["MODEL_CACHE_DIR"] = str(_default_host_cache())

    mc = _import_cache()
    mc.configure()
    return mc.cache_root()


def model_rows(spec) -> list[ModelRow]:
    mc = _import_cache()
    return [
        ModelRow(
            task="transcribe",
            label=f"{spec.transcribe_model} + align ({spec.transcribe_align_lang})",
            cached=mc.whisper_model_cached(spec.transcribe_model),
        ),
        ModelRow(
            task="translate",
            label=spec.translate_model,
            cached=mc.hf_model_cached(spec.translate_model),
        ),
        ModelRow(
            task="tts",
            label=f"{spec.tts_repo} (lang={spec.tts_lang})",
            cached=mc.hf_model_cached(spec.tts_repo),
        ),
    ]


def download_transcribe(*, model: str, device: str, align_lang: str) -> StepResult:
    from faster_whisper import WhisperModel
    import whisperx

    try:
        from models.whisper import whisper_hf_repo
    except ImportError:
        from whisper import whisper_hf_repo

    mc = _import_cache()
    compute_type = "float16" if device == "cuda" else "int8"
    repo = whisper_hf_repo(model)
    local_only = mc.whisper_model_cached(model)

    if not local_only:
        WhisperModel(
            model,
            device=device,
            compute_type=compute_type,
            download_root=str(mc.faster_whisper_root()),
            local_files_only=False,
        )

    whisperx.load_align_model(
        language_code=align_lang,
        device=device,
        model_dir=str(mc.hf_hub_cache()),
    )

    return StepResult(
        task="transcribe",
        outcome=StepOutcome.CACHED if local_only else StepOutcome.DOWNLOADED,
        model=f"{model} ({repo})",
        detail=f"align={align_lang}, device={device}",
    )


def download_translate(*, model: str) -> StepResult:
    from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

    mc = _import_cache()
    cache = str(mc.hf_hub_cache())
    local_only = mc.hf_model_cached(model)

    NllbTokenizer.from_pretrained(
        model, src_lang="eng_Latn", cache_dir=cache, local_files_only=local_only
    )
    AutoModelForSeq2SeqLM.from_pretrained(
        model, cache_dir=cache, local_files_only=local_only
    )

    return StepResult(
        task="translate",
        outcome=StepOutcome.CACHED if local_only else StepOutcome.DOWNLOADED,
        model=model,
    )


def download_tts(*, lang: str, repo: str) -> StepResult:
    mc = _import_cache()
    mc.configure()
    from kokoro import KPipeline

    local_only = mc.hf_model_cached(repo)
    KPipeline(lang_code=lang)

    return StepResult(
        task="tts",
        outcome=StepOutcome.CACHED if local_only else StepOutcome.DOWNLOADED,
        model=f"{repo} (lang={lang})",
    )


def _run_task(label: TaskName, fn, /, **kwargs) -> StepResult:
    try:
        return fn(**kwargs)
    except ImportError as exc:
        return StepResult(
            task=label,
            outcome=StepOutcome.SKIPPED,
            model=str(kwargs.get("model", label)),
            detail=str(exc),
        )
    except Exception as exc:
        return StepResult(
            task=label,
            outcome=StepOutcome.FAILED,
            model=str(kwargs.get("model", label)),
            detail=str(exc),
        )


def run_downloads(
    tasks: list[TaskName],
    *,
    models_dir: Path | None,
    device: str,
) -> tuple[Path, list[StepResult]]:
    mc = _import_cache()
    root = ensure_cache_dir(models_dir)
    spec = mc.load_model_spec()
    results: list[StepResult] = []

    if "transcribe" in tasks:
        results.append(
            _run_task(
                "transcribe",
                download_transcribe,
                model=spec.transcribe_model,
                device=device,
                align_lang=spec.transcribe_align_lang,
            )
        )
    if "translate" in tasks:
        results.append(_run_task("translate", download_translate, model=spec.translate_model))
    if "tts" in tasks:
        results.append(_run_task("tts", download_tts, lang=spec.tts_lang, repo=spec.tts_repo))

    return root, results


def _main_argparse(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download pipeline models to the shared cache")
    parser.add_argument(
        "task",
        choices=("all", "transcribe", "translate", "tts", "status"),
        nargs="?",
        default="all",
        help="Which model set to download, or show cache status (default: all)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Cache root (default: MODEL_CACHE_DIR or ./data/models)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device for transcribe download (default: cpu)",
    )
    args = parser.parse_args(argv)

    mc = _import_cache()
    root = ensure_cache_dir(args.models_dir)
    spec = mc.load_model_spec()

    print(f"Model cache: {root}", flush=True)
    print(
        f"Models: transcribe={spec.transcribe_model} translate={spec.translate_model} "
        f"tts={spec.tts_voice}/{spec.tts_lang}",
        flush=True,
    )

    if args.task == "status":
        for row in model_rows(spec):
            state = "cached" if row.cached else "missing"
            print(f"  {row.task:10} {state:7}  {row.label}", flush=True)
        return

    tasks: list[TaskName] = (
        ["transcribe", "translate", "tts"]
        if args.task == "all"
        else [args.task]  # type: ignore[list-item]
    )
    _, results = run_downloads(tasks, models_dir=args.models_dir, device=args.device)
    for result in results:
        if result.outcome == StepOutcome.CACHED:
            print(f"SKIP (cached): {result.task} {result.model}", flush=True)
        elif result.outcome == StepOutcome.SKIPPED:
            print(f"SKIP {result.task} (not available): {result.detail}", flush=True)
        elif result.outcome == StepOutcome.DOWNLOADED:
            print(f"Downloaded {result.task}: {result.model}", flush=True)
        elif result.outcome == StepOutcome.FAILED:
            print(f"FAILED {result.task}: {result.detail}", flush=True)
    print("Done.", flush=True)


def main(argv: list[str] | None = None) -> None:
    try:
        import typer  # noqa: F401
    except ImportError:
        _main_argparse(argv)
        return

    from models.download_cli import run_cli

    run_cli(argv)


if __name__ == "__main__":
    main()
