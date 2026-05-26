"""
Trigger one batched video-dubbing run.

Usage:
    python -m hatchet.trigger run sample.mp4 --run-id demo
    python -m hatchet.trigger run sample_batch/ --run-id batch-001
    python -m hatchet.trigger run sample_batch/ --run-id batch-001 --force
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer

from pipeline.config import get_config
from pipeline.paths import resolve_video_keys
from pipeline.utils import print_run_summary

if TYPE_CHECKING:
    from pipeline.run import PipelineRun

app = typer.Typer(
    name="trigger",
    help="Trigger a batched video-dubbing Hatchet workflow run.",
    no_args_is_help=True,
)


@app.callback()
def _root() -> None:
    """Hatchet batch dubbing trigger."""


def _print_summary(run: PipelineRun, video_keys: list[str], run_url: str | None = None) -> None:
    footer = "\n[bold green]Run started![/bold green]"
    if run_url:
        footer += f"\nWorkflow ID: {run_url}"
    footer += "\n[link=https://cloud.hatchet.run]https://cloud.hatchet.run[/link]"

    print_run_summary(
        run, video_keys,
        title="Triggering pipeline",
        border_style="blue",
        extra_rows=[("Batch ID", run.batch_id)],
        footer=footer,
    )


def run_trigger(
    video_keys: list[str],
    *,
    lang: str,
    run_id: str,
    batch_id: str,
    force: bool,
) -> str:
    from hatchet.workflow import workflow
    from pipeline.run import PipelineRun

    if not video_keys:
        raise typer.BadParameter("No video inputs to process")

    run_input = PipelineRun(
        video_keys=video_keys,
        target_lang=lang,
        run_id=run_id,
        batch_id=batch_id,
        force=force,
    )
    ref = workflow.run_no_wait(run_input)
    _print_summary(run_input, video_keys, run_url=ref.workflow_run_id)
    return ref.workflow_run_id


@app.command("run")
def cmd_run(
    source: str = typer.Argument(
        ...,
        help="Video file or folder prefix under the bucket mount (e.g. sample.mp4 or sample_batch/)",
    ),
    lang: str = typer.Option(get_config().pipeline.target_lang, "--lang", help="Target language code"),
    run_id: str = typer.Option("demo", "--run-id", help="Run label (output namespace)"),
    batch_id: str = typer.Option(
        "default", "--batch-id", help="Hatchet concurrency key for parallel batch runs"
    ),
    force: bool = typer.Option(
        False, "--force", help="Reprocess every file even when outputs already exist"
    ),
) -> None:
    """Trigger dubbing for one video or all videos under a prefix."""
    try:
        video_keys = resolve_video_keys(source)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    run_trigger(video_keys, lang=lang, run_id=run_id, batch_id=batch_id, force=force)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
