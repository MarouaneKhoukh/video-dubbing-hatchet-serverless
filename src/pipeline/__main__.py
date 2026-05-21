"""
Local pipeline CLI — run dubbing via Docker on ./data (no Hatchet/Nebius).

Usage:
    python -m pipeline run sample.mp4 --run-id demo
    python -m pipeline run sample_batch/ --run-id batch-001
    python -m pipeline run sample.mp4 --stage transcribe --run-id demo
    python -m pipeline run sample.mp4 --force --device cpu
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import typer
from rich.panel import Panel
from rich.table import Table

from pipeline.config import get_settings
from pipeline.paths import repo_data_dir, resolve_video_keys
from pipeline.run import STAGE_ORDER, PipelineRun, run_pipeline, run_stage
from pipeline.runner.local import LocalExecutor
from pipeline.storage import use_local_artifacts
from pipeline.utils import get_console, setup_logging

app = typer.Typer(
    name="pipeline",
    help="Run the dubbing pipeline locally via Docker (./data mounted at /data).",
    no_args_is_help=True,
)
logger = logging.getLogger(__name__)


@app.callback()
def _root() -> None:
    """Local dubbing pipeline (Docker on ./data)."""


def _print_summary(
    run: PipelineRun,
    video_keys: list[str],
    *,
    data_dir: Path,
    stages: list[str],
    device: str,
    use_gpu: bool,
) -> None:
    console = get_console()
    mode = "single" if len(video_keys) == 1 else f"batch ({len(video_keys)} files)"

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="dim")
    table.add_column()
    table.add_row("Mode", mode)
    table.add_row("Executor", "local-docker")
    table.add_row("Data dir", str(data_dir))
    table.add_row("Language", run.target_lang)
    table.add_row("Run ID", run.run_id)
    table.add_row("Stages", ", ".join(stages))
    table.add_row("Force", str(run.force))
    table.add_row("Transcribe device", device)
    table.add_row("Docker GPU", str(use_gpu))
    table.add_row("Output prefix", f"runs/{run.run_id}/")

    if len(video_keys) <= 10:
        for vk in video_keys:
            table.add_row("Input", vk)
    else:
        for vk in video_keys[:5]:
            table.add_row("Input", vk)
        table.add_row("", f"… and {len(video_keys) - 5} more")

    console.print(Panel(table, title="Local pipeline run", border_style="green"))


@app.command("run")
def cmd_run(
    source: str = typer.Argument(
        ...,
        help="Video file or folder under data/ (e.g. sample.mp4 or sample_batch/)",
    ),
    lang: str = typer.Option(get_settings().target_lang, "--lang", help="Target language code"),
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
        help="Host data directory mounted at /data (default: <repo>/data)",
    ),
    models_dir: Path = typer.Option(
        None,
        "--models-dir",
        help="Model cache directory (default: <data-dir>/models)",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="Transcribe device passed to the container (cpu or cuda)",
    ),
    gpus: bool = typer.Option(
        False,
        "--gpus",
        help="Pass --gpus all to GPU task containers (transcribe, tts)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    """Run dubbing locally for one video or all videos under a prefix."""
    setup_logging(verbose=verbose)
    resolved_data = (data_dir or repo_data_dir()).resolve()
    resolved_models = (models_dir or resolved_data / "models").resolve()

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
    executor = LocalExecutor(resolved_data, models_dir=resolved_models, use_gpu=gpus)
    _print_summary(
        run_input,
        video_keys,
        data_dir=resolved_data,
        stages=stages,
        device=device,
        use_gpu=gpus,
    )

    cli_overrides = {"transcribe": {"device": device}}

    async def _run() -> dict[str, dict]:
        with use_local_artifacts(resolved_data):
            if len(stages) == 1:
                result = await run_stage(
                    stages[0],
                    run_input,
                    executor=executor,
                    cli_overrides=cli_overrides,
                    log=logger.info,
                )
                return {stages[0]: result}
            return await run_pipeline(
                run_input,
                executor=executor,
                stages=stages,
                cli_overrides=cli_overrides,
                log=logger.info,
            )

    results = asyncio.run(_run())
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
