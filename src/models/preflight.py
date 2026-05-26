"""Model-presence pre-flight check — single function shared by local + cloud paths.

Two callers:
  - ``pipeline.run.py`` (local mode) — checks the on-disk model cache (``~/.cache/hf``,
    ``data/models/``). Warn-only: HF Hub auto-downloads on first use.
  - ``hatchet.workflow.py`` (cloud mode) — checks the Nebius bucket via S3 list.
    Hard fail: cold-start auto-download from HF fails on FUSE (talk.md bug #5).

Single source of truth for "what models does each stage need" — read from
``get_config().pipeline``, so changing a model ID in config.py automatically
updates the pre-flight requirements, the sync script, and the upload list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pipeline.config import get_config

from models.model_cache import hf_model_cached, whisper_model_cached

try:
    from models.whisper import whisper_hf_repo
except ImportError:
    from whisper import whisper_hf_repo


# Bucket root for all model objects. Matches the FUSE-mounted path the GPU
# Dockerfiles point HF_HOME / MODEL_CACHE_DIR at (/data/models/...).
BUCKET_MODEL_ROOT = "models"

Location = Literal["local", "remote"]


@dataclass(frozen=True)
class ModelRef:
    """One model the pipeline depends on. ``kind`` picks the cache layout:
    - ``"whisper"`` → faster-whisper cache (``models/faster-whisper/models--<repo>/``)
    - ``"hf"``      → HF Hub cache         (``models/huggingface/hub/models--<repo>/``)
    """
    kind: Literal["whisper", "hf"]
    id: str


def _expected_for(stage: str) -> list[ModelRef]:
    """Models this stage reads at runtime. Empty for CPU-only stages."""
    cfg = get_config().pipeline
    if stage == "transcribe":
        return [ModelRef("whisper", cfg.transcribe.model)]
    if stage == "translate":
        return [ModelRef("hf", cfg.translate.model)]
    if stage == "tts":
        return [ModelRef("hf", cfg.tts.repo)]
    return []


def _hf_repo_dir(repo_id: str) -> str:
    """HF Hub's on-disk repo dirname: ``org/name`` → ``models--org--name``."""
    return f"models--{repo_id.replace('/', '--')}"


def _bucket_prefix(ref: ModelRef) -> str:
    """S3 prefix where this model's blobs live in the Nebius bucket."""
    if ref.kind == "whisper":
        return f"{BUCKET_MODEL_ROOT}/faster-whisper/{_hf_repo_dir(whisper_hf_repo(ref.id))}/"
    return f"{BUCKET_MODEL_ROOT}/huggingface/hub/{_hf_repo_dir(ref.id)}/"


def _present(ref: ModelRef, location: Location) -> bool:
    """Does this model exist at the requested location?"""
    if location == "local":
        if ref.kind == "whisper":
            return whisper_model_cached(ref.id)
        return hf_model_cached(ref.id)
    if location == "remote":
        from pipeline.storage import list_objects
        return bool(list_objects(_bucket_prefix(ref)))
    raise ValueError(f"unknown location {location!r}; expected 'local' or 'remote'")


def pre_flight_check(stage: str, location: Location) -> tuple[bool, list[str]]:
    """Return ``(all_present, missing_ids)`` for *stage* at *location*.

    No side effects. The caller decides what to do with the result:
      - workflow.py: raise on missing remote — HF auto-download is broken on FUSE.
      - run.py:      log a warning on missing local — HF will auto-download on use.

    Returns an empty ``missing`` list (and ``True``) for stages with no model
    deps (extract, remux).
    """
    expected = _expected_for(stage)
    if not expected:
        return True, []
    missing = [ref.id for ref in expected if not _present(ref, location)]
    return not missing, missing
