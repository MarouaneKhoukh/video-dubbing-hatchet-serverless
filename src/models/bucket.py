"""Upload the local model cache to the Nebius bucket.

Operator-side helper, used by ``scripts/sync_models.py``. Not called by the
Hatchet workflow or by container jobs.

The bucket layout matches the on-disk cache layout under ``MODEL_CACHE_DIR``,
so containers see the same paths via the FUSE mount at ``/data/models/...``.
Bucket prefix mapping lives in ``models.preflight._bucket_prefix`` (single
source of truth).
"""

from __future__ import annotations

import logging
from pathlib import Path

from pipeline.storage import list_objects, upload_to_storage

from models.model_cache import cache_root, configure
from models.preflight import (
    BUCKET_MODEL_ROOT,
    _bucket_prefix,
    _expected_for,
)

logger = logging.getLogger(__name__)


def _iter_local_files_under(prefix: str) -> list[Path]:
    """Walk the local mirror of *prefix* (a bucket-relative path) and yield files."""
    configure()  # ensure MODEL_CACHE_DIR / HF_HOME etc. are set
    local_sub = prefix.removeprefix(BUCKET_MODEL_ROOT + "/")
    base = cache_root() / local_sub
    if not base.is_dir():
        return []
    return sorted(p for p in base.rglob("*") if p.is_file())


def upload_models_to_bucket(stages: list[str]) -> dict[str, int]:
    """Mirror the local model cache for each stage → ``s3://<bucket>/models/...``.

    Skips files whose bucket key already exists (idempotent — safe to re-run).
    Returns ``{stage: files_uploaded}``.
    """
    configure()
    root = cache_root()
    uploaded_per_stage: dict[str, int] = {}

    for stage in stages:
        uploaded = 0
        for ref in _expected_for(stage):
            bucket_prefix = _bucket_prefix(ref)
            existing = set(list_objects(bucket_prefix))
            local_files = _iter_local_files_under(bucket_prefix)
            if not local_files:
                logger.warning(
                    f"[{stage}] no local files for {ref.id} under {root}; "
                    f"download it first"
                )
                continue
            for local_path in local_files:
                rel = local_path.relative_to(root)
                bucket_key = f"{BUCKET_MODEL_ROOT}/{rel.as_posix()}"
                if bucket_key in existing:
                    continue
                size = local_path.stat().st_size
                logger.info(f"[{stage}] uploading {bucket_key} ({size} bytes)")
                upload_to_storage(local_path, bucket_key)
                uploaded += 1
        uploaded_per_stage[stage] = uploaded

    return uploaded_per_stage
