#!/usr/bin/env python3
"""Sync pipeline models: local cache → Nebius bucket. One-shot, idempotent.

Flow:
    1. Check the bucket for each GPU stage's required models.
    2. For stages that are missing in the bucket, download them locally (HF Hub).
    3. Upload the local cache → bucket for those stages.
    4. Verify post-upload.

Safe to re-run: stages already present in the bucket are skipped entirely.
Cannot be parameterised; runs all three GPU stages by design (single,
predictable flow).
"""

from __future__ import annotations

import logging

from models.bucket import upload_models_to_bucket
from models.download import run_downloads
from models.preflight import pre_flight_check

STAGES = ["transcribe", "translate", "tts"]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # 1. What's already in the bucket?
    print("== Checking bucket ==")
    missing_per_stage: dict[str, list[str]] = {}
    for stage in STAGES:
        present, missing = pre_flight_check(stage, location="remote")
        mark = "OK" if present else "MISS"
        info = "present" if present else f"missing {missing}"
        print(f"  [{mark:4}] {stage}: {info}")
        if not present:
            missing_per_stage[stage] = missing

    if not missing_per_stage:
        print("\n✓ Bucket already has all expected models. Nothing to do.")
        return 0

    stages_to_sync = list(missing_per_stage)

    # 2. Download locally what's missing (HF Hub fills the on-disk cache).
    print(f"\n== Downloading locally (stages: {', '.join(stages_to_sync)}) ==")
    run_downloads(stages_to_sync)

    # 3. Upload to bucket.
    print("\n== Uploading to bucket ==")
    uploaded = upload_models_to_bucket(stages_to_sync)
    for stage, n in uploaded.items():
        print(f"  {stage}: {n} file(s) uploaded")

    # 4. Verify.
    print("\n== Post-upload verification ==")
    failed: list[tuple[str, list[str]]] = []
    for stage in stages_to_sync:
        present, missing = pre_flight_check(stage, location="remote")
        mark = "OK" if present else "FAIL"
        print(f"  [{mark:4}] {stage}")
        if not present:
            failed.append((stage, missing))

    if failed:
        print("\n✗ Verification failed:")
        for stage, missing in failed:
            print(f"  {stage}: still missing {missing}")
        return 1

    print("\n✓ All stages verified in bucket.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
