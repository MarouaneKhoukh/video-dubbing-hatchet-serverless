"""
Start the Hatchet worker.

Usage:
    python worker.py

The worker connects to Hatchet (via HATCHET_CLIENT_TOKEN) and waits for runs.
Keep this running while you trigger pipeline runs from trigger.py.
"""

import logging

from pipeline.workflow import (
    hatchet,
    workflow,
    extract_audio,
    transcribe_whisper,
    translate_text_step,
    synthesize_tts,
    remux_video,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main():
    worker = hatchet.worker("dubbing-worker")
    worker.register_workflow(workflow)
    print("Worker started. Waiting for runs...")
    print("Open the Hatchet dashboard: https://cloud.onhatchet.run")
    worker.start()


if __name__ == "__main__":
    main()
