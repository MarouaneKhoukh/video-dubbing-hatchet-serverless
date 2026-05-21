"""Typer + Rich CLI for model pre-download (host only)."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.panel import Panel
from rich.table import Table

from models.console import get_console, setup_logging
from models.download_models import (
    StepOutcome,
    StepResult,
    TaskName,
    _default_host_cache,
    _import_cache,
    ensure_cache_dir,
    model_rows,
    run_downloads,
)

app = typer.Typer(
    name="download-models",
    help="Pre-download pipeline models to the shared cache (data/models).",
    no_args_is_help=True,
)

console = get_console()


def _resolve_options(
    models_dir: Path | None,
    verbose: bool,
) -> tuple[Path, object]:
    setup_logging(verbose=verbose)
    mc = _import_cache()
    root = ensure_cache_dir(models_dir)
    spec = mc.load_model_spec()
    return root, spec


def _print_header(root: Path, spec) -> None:
    lines = [
        f"[bold]Cache[/bold]  {root}",
        f"[bold]ASR[/bold]    {spec.transcribe_model} (align={spec.transcribe_align_lang})",
        f"[bold]NLLB[/bold]   {spec.translate_model}",
        f"[bold]TTS[/bold]    {spec.tts_voice} / {spec.tts_lang}",
    ]
    console.print(Panel("\n".join(lines), title="Model cache", border_style="blue"))


def _status_table(rows) -> Table:
    table = Table(title="Cache status", show_header=True, header_style="bold")
    table.add_column("Task", style="cyan")
    table.add_column("Model")
    table.add_column("Status", justify="center")
    for row in rows:
        status = "[green]cached[/green]" if row.cached else "[yellow]missing[/yellow]"
        table.add_row(row.task, row.label, status)
    return table


def _outcome_label(result: StepResult) -> str:
    match result.outcome:
        case StepOutcome.CACHED:
            return "[green]cached[/green]"
        case StepOutcome.DOWNLOADED:
            return "[green]downloaded[/green]"
        case StepOutcome.SKIPPED:
            return f"[yellow]skipped[/yellow] ({result.detail})"
        case StepOutcome.FAILED:
            return f"[red]failed[/red] ({result.detail})"
    return result.outcome.value


def _print_results(results: list[StepResult]) -> None:
    table = Table(title="Results", show_header=True, header_style="bold")
    table.add_column("Task", style="cyan")
    table.add_column("Model")
    table.add_column("Outcome")
    for result in results:
        table.add_row(result.task, result.model, _outcome_label(result))
    console.print(table)


def _finish_summary(results: list[StepResult]) -> None:
    downloaded = sum(1 for r in results if r.outcome == StepOutcome.DOWNLOADED)
    cached = sum(1 for r in results if r.outcome == StepOutcome.CACHED)
    skipped = sum(1 for r in results if r.outcome == StepOutcome.SKIPPED)
    failed = sum(1 for r in results if r.outcome == StepOutcome.FAILED)
    lines = [
        f"[green]✓[/green] {cached} already cached",
        f"[green]✓[/green] {downloaded} downloaded",
    ]
    if skipped:
        lines.append(f"[yellow]![/yellow] {skipped} skipped (deps missing in this env)")
    if failed:
        lines.append(f"[red]✗[/red] {failed} failed")
    border = "red" if failed else "green"
    console.print(Panel("\n".join(lines), title="Done", border_style=border))
    if failed:
        raise typer.Exit(code=1)


def _download(
    tasks: list[TaskName],
    *,
    models_dir: Path | None,
    device: str,
    verbose: bool,
) -> None:
    root, spec = _resolve_options(models_dir, verbose)
    _print_header(root, spec)
    try:
        _, results = run_downloads(tasks, models_dir=models_dir, device=device)
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    _print_results(results)
    _finish_summary(results)


@app.command("status")
def cmd_status(
    models_dir: Path | None = typer.Option(
        None,
        "--models-dir",
        help=f"Cache root (default: {_default_host_cache()})",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    """Show which models are already in the cache."""
    root, spec = _resolve_options(models_dir, verbose)
    _print_header(root, spec)
    console.print(_status_table(model_rows(spec)))


@app.command("all")
def cmd_all(
    models_dir: Path | None = typer.Option(None, "--models-dir", help="Cache root"),
    device: str = typer.Option("cpu", "--device", help="Device for transcribe weights"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    """Download transcribe, translate, and TTS models."""
    _download(
        ["transcribe", "translate", "tts"],
        models_dir=models_dir,
        device=device,
        verbose=verbose,
    )


@app.command("transcribe")
def cmd_transcribe(
    models_dir: Path | None = typer.Option(None, "--models-dir", help="Cache root"),
    device: str = typer.Option("cpu", "--device", help="cpu or cuda"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    """Download faster-whisper + WhisperX align weights."""
    _download(["transcribe"], models_dir=models_dir, device=device, verbose=verbose)


@app.command("translate")
def cmd_translate(
    models_dir: Path | None = typer.Option(None, "--models-dir", help="Cache root"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    """Download NLLB tokenizer and model."""
    _download(["translate"], models_dir=models_dir, device="cpu", verbose=verbose)


@app.command("tts")
def cmd_tts(
    models_dir: Path | None = typer.Option(None, "--models-dir", help="Cache root"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    """Download Kokoro TTS weights."""
    _download(["tts"], models_dir=models_dir, device="cpu", verbose=verbose)


def run_cli(argv: list[str] | None = None) -> None:
    if argv is None:
        app()
        return
    import typer.main

    typer.main.get_command(app)(prog_name="download_models", args=argv, standalone_mode=False)
