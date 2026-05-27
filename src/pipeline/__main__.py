"""
Local pipeline CLI — run dubbing stages in-process Python (no Docker/Hatchet).

Usage:
    python -m pipeline run sample_file/sample.mp4 --run-id demo
    python -m pipeline run sample_batch/           --run-id batch-001
    python -m pipeline run sample_file/sample.mp4 --stage transcribe --run-id demo
    python -m pipeline run sample_file/sample.mp4 --force --device cpu

    python -m pipeline prepare-manifest extract    --run-id demo --source sample_file/sample.mp4
    python -m pipeline prepare-manifest transcribe --run-id demo
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.panel import Panel

from pipeline.config import get_config
from pipeline.paths import (
    repo_data_dir,
    resolve_video_keys,
    task_manifest_key,
)
from pipeline.run import STAGE_ORDER, PipelineRun, run_pipeline, run_stage
from pipeline.storage import use_local_artifacts
from pipeline.utils import get_console, print_run_summary, setup_logging

app = typer.Typer(
    name="pipeline",
    help="Run the dubbing pipeline locally in-process Python (./data as the artifact root).",
    no_args_is_help=True,
)
logger = logging.getLogger(__name__)


@app.callback()
def _root() -> None:
    """Local dubbing pipeline (Python in-process)."""


def _print_summary(
    run: PipelineRun,
    video_keys: list[str],
    *,
    data_dir: Path,
    stages: list[str],
    device: str,
) -> None:
    print_run_summary(
        run, video_keys,
        title="Local pipeline run",
        border_style="green",
        extra_rows=[
            ("Executor", "python (in-process)"),
            ("Data dir", str(data_dir)),
            ("Stages", ", ".join(stages)),
            ("Transcribe device", device),
        ],
    )


@app.command("run")
def cmd_run(
    source: str = typer.Argument(
        ...,
        help="Video file or folder under data/ (e.g. sample_file/sample.mp4 or sample_batch/)",
    ),
    lang: str = typer.Option(get_config().pipeline.target_lang, "--lang", help="Target language code"),
    run_id: str = typer.Option("demo", "--run-id", help="Run label (output namespace)"),
    batch_id: str = typer.Option("local", "--batch-id", help="Batch label (stored on run manifest)"),
    force: bool = typer.Option(
        False, "--force", help="Reprocess every file even when outputs already exist"
    ),
    stage: list[str] = typer.Option(
        None,
        "--stage",
        help=f"Run only these stage(s); repeat flag for multiple. Default: all ({', '.join(STAGE_ORDER)})",
    ),
    data_dir: Path = typer.Option(
        None,
        "--data-dir",
        help="Host data directory used as artifact root (default: <repo>/data)",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="Device override for transcribe/translate/tts (cpu or cuda)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    """Run dubbing locally for one video or all videos under a prefix."""
    setup_logging(verbose=verbose)
    resolved_data = (data_dir or repo_data_dir()).resolve()

    try:
        with use_local_artifacts(resolved_data):
            video_keys = resolve_video_keys(source, data_root=resolved_data)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if not video_keys:
        raise typer.BadParameter("No video inputs to process")

    stages = list(stage) if stage else list(STAGE_ORDER)
    unknown = [s for s in stages if s not in STAGE_ORDER]
    if unknown:
        raise typer.BadParameter(
            f"Unknown stage(s) {unknown}; choose from: {', '.join(STAGE_ORDER)}"
        )

    run_input = PipelineRun(
        video_keys=video_keys,
        target_lang=lang,
        run_id=run_id,
        batch_id=batch_id,
        force=force,
    )
    _print_summary(
        run_input,
        video_keys,
        data_dir=resolved_data,
        stages=stages,
        device=device,
    )

    cli_overrides = {
        task: {"device": device} for task in ("transcribe", "translate", "tts")
    }

    with use_local_artifacts(resolved_data):
        if len(stages) == 1:
            results = {
                stages[0]: run_stage(
                    stages[0],
                    run_input,
                    cli_overrides=cli_overrides,
                    log=logger.info,
                )
            }
        else:
            results = run_pipeline(
                run_input,
                stages=stages,
                cli_overrides=cli_overrides,
                log=logger.info,
            )

    get_console().print(
        Panel(
            f"[bold green]Done.[/bold green] Outputs under [bold]{resolved_data}/runs/{run_id}/[/bold]\n"
            f"Manifests: [bold]{resolved_data}/runs/{run_id}/manifests/[/bold]  "
            f"Reports: [bold]{resolved_data}/runs/{run_id}/reports/[/bold]",
            border_style="green",
        )
    )
    if verbose:
        get_console().print(results)


@app.command("prepare-manifest")
def cmd_prepare_manifest(
    stage: str = typer.Argument(..., help=f"Stage name: one of {', '.join(STAGE_ORDER)}"),
    run_id: str = typer.Option(..., "--run-id", help="Run label (output namespace)"),
    source: str = typer.Option(
        None,
        "--source",
        help="Video file/folder under data/ (required for extract; ignored for downstream stages)",
    ),
    lang: str = typer.Option(get_config().pipeline.target_lang, "--lang", help="Target language code"),
    batch_id: str = typer.Option("local", "--batch-id", help="Batch label"),
    force: bool = typer.Option(False, "--force", help="Set force=true in the manifest"),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="Device override for transcribe/translate/tts (cpu or cuda)",
    ),
    data_dir: Path = typer.Option(
        None, "--data-dir", help="Host data directory (default: <repo>/data)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    """Write a task manifest to runs/<id>/manifests/<stage>.json (no execution)."""
    from pipeline.metadata import build_task_manifest
    from pipeline.storage import upload_json

    setup_logging(verbose=verbose)
    if stage not in STAGE_ORDER:
        raise typer.BadParameter(f"Unknown stage {stage!r}; choose from: {', '.join(STAGE_ORDER)}")

    resolved_data = (data_dir or repo_data_dir()).resolve()

    if stage == "extract":
        if not source:
            raise typer.BadParameter("--source is required for the extract stage")
        try:
            with use_local_artifacts(resolved_data):
                video_keys = resolve_video_keys(source, data_root=resolved_data)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    else:
        video_keys = []  # downstream stages derive inputs from upstream reports

    run_input = PipelineRun(
        video_keys=video_keys,
        target_lang=lang,
        run_id=run_id,
        batch_id=batch_id,
        force=force,
    )
    cli_overrides = {
        task: {"device": device} for task in ("transcribe", "translate", "tts")
    }

    with use_local_artifacts(resolved_data):
        manifest = build_task_manifest(
            run_input, stage, cli_overrides=cli_overrides, executor="docker"
        )
        key = task_manifest_key(run_id, stage)
        upload_json(manifest.model_dump(), key)

    get_console().print(
        Panel(
            f"Manifest written: [bold]{resolved_data / key}[/bold]\n"
            f"Container path:   [bold]/data/{key}[/bold]",
            title=f"prepare-manifest {stage}",
            border_style="green",
        )
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
