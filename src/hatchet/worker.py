"""
Start the Hatchet worker.

Usage:
    python -m hatchet.worker

The worker connects to Hatchet (via HATCHET_CLIENT_TOKEN) and waits for batch
runs (`video-dubbing-batch-pipeline`). Keep this running while you trigger
pipeline runs from `python -m hatchet.trigger`.
"""

from __future__ import annotations

import logging

from rich.panel import Panel

from hatchet.console import get_console, setup_logging
from hatchet.workflow import hatchet, workflow

logger = logging.getLogger(__name__)


def run_worker() -> None:
    """Register the batch workflow and block until interrupted."""
    setup_logging(verbose=False)
    console = get_console()
    console.print(
        Panel(
            "[bold]video-dubbing-batch-pipeline[/bold]\n"
            "Waiting for batch runs…\n"
            "[link=https://cloud.hatchet.run]Hatchet dashboard[/link]",
            title="Dubbing worker",
            border_style="cyan",
        )
    )
    worker = hatchet.worker("dubbing-batch-worker")
    worker.register_workflow(workflow)
    logger.info("Worker registered; listening for events")
    worker.start()


def main() -> None:
    run_worker()


if __name__ == "__main__":
    main()
