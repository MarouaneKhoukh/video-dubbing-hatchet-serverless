"""Small shared helpers for the pipeline package."""

from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Sequence

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from pipeline.run import PipelineRun

_console: Console | None = None


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def get_console() -> Console:
    global _console
    if _console is None:
        _console = Console(stderr=True)
    return _console


def print_run_summary(
    run: PipelineRun,
    video_keys: list[str],
    *,
    title: str,
    border_style: str = "blue",
    extra_rows: Sequence[tuple[str, str]] = (),
    footer: str | None = None,
) -> None:
    """Render a Rich-style run banner. Shared by L1/L2 CLI and the Hatchet trigger.

    Common rows (Mode / Language / Run ID / Force / Output prefix / Input listing
    with truncation) are emitted unconditionally. ``extra_rows`` are inserted
    immediately after "Mode" — that's where each caller adds its own context
    (Executor + Data dir + Stages for local; Batch ID for Hatchet).
    """
    console = get_console()
    mode = "single" if len(video_keys) == 1 else f"batch ({len(video_keys)} files)"

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="dim")
    table.add_column()
    table.add_row("Mode", mode)
    for label, value in extra_rows:
        table.add_row(label, value)
    table.add_row("Language", run.target_lang)
    table.add_row("Run ID", run.run_id)
    table.add_row("Force", str(run.force))
    table.add_row("Output prefix", f"runs/{run.run_id}/")

    if len(video_keys) <= 10:
        for vk in video_keys:
            table.add_row("Input", vk)
    else:
        for vk in video_keys[:5]:
            table.add_row("Input", vk)
        table.add_row("", f"… and {len(video_keys) - 5} more")

    console.print(Panel(table, title=title, border_style=border_style))
    if footer:
        console.print(footer)


def setup_logging(*, verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler]
    if sys.stderr.isatty():
        handlers = [
            RichHandler(
                console=get_console(),
                rich_tracebacks=True,
                show_time=False,
                show_path=verbose,
            )
        ]
    else:
        handlers = [logging.StreamHandler(sys.stderr)]
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        handlers=handlers,
        force=True,
    )
