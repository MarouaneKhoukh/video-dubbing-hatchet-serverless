"""Download sample videos for local / batch pipeline testing."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import requests
import typer
from rich.panel import Panel
from rich.progress import BarColumn, DownloadColumn, Progress, TransferSpeedColumn

from hatchet.console import get_console, setup_logging

app = typer.Typer(
    name="download",
    help="Download sample videos to data/ for pipeline testing.",
    no_args_is_help=True,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
SAMPLE_NAME = "sample.mp4"
BATCH_DIR_NAME = "sample_batch"
MANIFEST_NAME = "manifest.txt"

NASA_CORONAGRAPH_2MIN_URL = (
    "https://assets.science.nasa.gov/content/dam/science/astro/"
    "programs/exep/technology/videos/coronagraph_2min.mp4"
)
TEARS_OF_STEEL_YOUTUBE_URL = "https://www.youtube.com/watch?v=R6MlUcmOul8"

logger = logging.getLogger(__name__)


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Required tool not found on PATH: {name}")
    return path


def _run(cmd: Sequence[str]) -> None:
    logger.debug("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, text=True)


def _download_url(url: str, dest: Path, *, timeout: float = 120.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    console = get_console()
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0)) or None
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(f"Downloading {dest.name}", total=total)
            with dest.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        handle.write(chunk)
                        progress.update(task, advance=len(chunk))


def _trim_video(src: Path, dest: Path, *, duration: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.warning("ffmpeg not found — copying full file to %s", dest.name)
        shutil.copy2(src, dest)
        return
    _run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(src),
            "-t",
            str(duration),
            "-c",
            "copy",
            str(dest),
        ]
    )


def download_nasa_sample(data_dir: Path, *, duration: float) -> Path:
    """NASA coronagraph narrated clip (public domain) → data/sample.mp4."""
    sample = data_dir / SAMPLE_NAME
    with tempfile.TemporaryDirectory(prefix="nasa-sample-") as tmp:
        raw = Path(tmp) / "coronagraph_2min.mp4"
        trimmed = Path(tmp) / "trimmed.mp4"
        _download_url(NASA_CORONAGRAPH_2MIN_URL, raw)
        _trim_video(raw, trimmed, duration=duration)
        sample.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(trimmed, sample)
    logger.info("Sample ready: %s (%.0fs)", sample, duration)
    return sample


def download_tears_of_steel_sample(
    data_dir: Path,
    *,
    start: int,
    duration: int,
) -> Path:
    """Blender *Tears of Steel* segment via YouTube (CC BY) → data/sample.mp4."""
    _require_tool("yt-dlp")
    sample = data_dir / SAMPLE_NAME
    with tempfile.TemporaryDirectory(prefix="tos-sample-") as tmp:
        clip = Path(tmp) / "clip.mp4"
        end = start + duration
        get_console().print(f"[dim]Fetching Tears of Steel via yt-dlp (t={start}s, {duration}s)…[/dim]")
        _run(
            [
                "yt-dlp",
                TEARS_OF_STEEL_YOUTUBE_URL,
                "--download-sections",
                f"*{start}-{end}",
                "-f",
                "bv*+ba/b",
                "--merge-output-format",
                "mp4",
                "-o",
                str(clip),
            ]
        )
        sample.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(clip, sample)
    logger.info("Sample ready: %s (%ds from t=%ds)", sample, duration, start)
    return sample


def populate_batch(sample: Path, data_dir: Path, count: int) -> list[Path]:
    """Copy sample into data/sample_batch/ as 001_sample.mp4, 002_sample.mp4, …"""
    if count < 1:
        raise ValueError("--sample-size must be >= 1")

    batch_dir = data_dir / BATCH_DIR_NAME
    batch_dir.mkdir(parents=True, exist_ok=True)

    width = max(3, len(str(count)))
    copies: list[Path] = []
    for i in range(1, count + 1):
        dest = batch_dir / f"{i:0{width}d}_{SAMPLE_NAME}"
        shutil.copy2(sample, dest)
        copies.append(dest)
        logger.info("Batch copy: %s", dest.relative_to(data_dir))

    manifest = batch_dir / MANIFEST_NAME
    keys = [f"{BATCH_DIR_NAME}/{path.name}" for path in copies]
    manifest.write_text("\n".join(keys) + "\n", encoding="utf-8")
    logger.info("Wrote manifest (%d keys): %s", len(keys), manifest.relative_to(data_dir))
    return copies


def _finish(data_dir: Path, sample: Path, sample_size: int | None) -> None:
    lines = [f"[green]✓[/green] Sample: [bold]{sample}[/bold]"]
    if sample_size is not None:
        manifest = data_dir / BATCH_DIR_NAME / MANIFEST_NAME
        lines.append(f"[green]✓[/green] Batch: {sample_size} copies under {data_dir / BATCH_DIR_NAME}")
        lines.append(f"[green]✓[/green] Manifest: {manifest}")
    get_console().print(Panel("\n".join(lines), title="Download complete", border_style="green"))


def _common_options(
    data_dir: Path,
    sample_size: int | None,
    verbose: bool,
) -> Path:
    setup_logging(verbose=verbose)
    resolved = data_dir.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


@app.command("nasa")
def cmd_nasa(
    duration: float = typer.Option(75.0, "--duration", help="Trim length in seconds"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir", help="Output directory"),
    sample_size: int | None = typer.Option(
        None, "--sample-size", help="Duplicate sample into sample_batch/"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    """NASA coronagraph narrated clip (public domain)."""
    data_dir = _common_options(data_dir, sample_size, verbose)
    try:
        sample = download_nasa_sample(data_dir, duration=duration)
        if sample_size is not None:
            populate_batch(sample, data_dir, sample_size)
        _finish(data_dir, sample, sample_size)
    except Exception as exc:
        get_console().print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command("tears-of-steel")
def cmd_tears_of_steel(
    start: int = typer.Option(300, "--start", help="Start time in seconds"),
    duration: int = typer.Option(60, "--duration", help="Clip length in seconds"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir", help="Output directory"),
    sample_size: int | None = typer.Option(
        None, "--sample-size", help="Duplicate sample into sample_batch/"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    """Blender Tears of Steel clip (CC BY, via yt-dlp)."""
    data_dir = _common_options(data_dir, sample_size, verbose)
    try:
        sample = download_tears_of_steel_sample(data_dir, start=start, duration=duration)
        if sample_size is not None:
            populate_batch(sample, data_dir, sample_size)
        _finish(data_dir, sample, sample_size)
    except Exception as exc:
        get_console().print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


def main() -> None:
    app()


if __name__ == "__main__":
    main()
    sys.exit(0)
