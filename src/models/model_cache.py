"""
Shared model cache for container jobs and local download.

Local dev:  host ``data/models`` → container ``/data/models`` (volume mount)
Nebius:     ``s3://<bucket>/models/`` → ``/data/models`` (bucket mounted at /data)

Model IDs default on ``pipeline.config`` task classes (container images ship ``/pipeline`` on ``PYTHONPATH``).
Set ``MODEL_CACHE_DIR`` to override the cache root. Unset on the host defaults to
``<repo>/data/models``; container images set ``MODEL_CACHE_DIR=/data/models``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _host_default_cache() -> Path:
    """Default cache when ``MODEL_CACHE_DIR`` is unset (host dev; container fallback)."""
    path = Path(__file__).resolve()
    # Container images copy this module to /model_cache.py (no repo tree).
    if path.parent == Path("/"):
        return Path("/data/models")
    # Repo layout: src/models/model_cache.py → <repo>/data/models
    return path.parents[2] / "data" / "models"


def config_module():
    """``pipeline.config`` on host and in container images (``/pipeline`` on ``PYTHONPATH``)."""
    try:
        from pipeline import config as cfg
    except ImportError:
        import pipeline_config as cfg  # legacy flat copy
    return cfg


@dataclass(frozen=True)
class ModelSpec:
    transcribe_model: str
    transcribe_device: str
    transcribe_align_lang: str
    translate_model: str
    tts_voice: str
    tts_lang: str
    tts_repo: str


def _task_defaults(cfg) -> ModelSpec:
    transcribe = cfg.TranscribeConfig()
    translate = cfg.TranslateConfig()
    tts = cfg.TtsConfig()
    return ModelSpec(
        transcribe_model=transcribe.model,
        transcribe_device=transcribe.device,
        transcribe_align_lang=transcribe.align_lang,
        translate_model=translate.model,
        tts_voice=tts.voice,
        tts_lang=tts.lang,
        tts_repo=tts.repo,
    )


def default_model_spec() -> ModelSpec:
    return _task_defaults(config_module())


def load_model_spec() -> ModelSpec:
    """Load from Settings when .env credentials exist; else task class defaults."""
    try:
        s = config_module().get_settings()
        return ModelSpec(
            transcribe_model=s.transcribe.model,
            transcribe_device=s.transcribe.device,
            transcribe_align_lang=s.transcribe.align_lang,
            translate_model=s.translate.model,
            tts_voice=s.tts.voice,
            tts_lang=s.tts.lang,
            tts_repo=s.tts.repo,
        )
    except Exception:
        return default_model_spec()


def cache_root() -> Path:
    if explicit := os.environ.get("MODEL_CACHE_DIR"):
        return Path(explicit)
    return _host_default_cache()


def configure() -> Path:
    """Create cache dirs and point HF / torch caches at the model volume."""
    root = cache_root()
    hf_home = root / "huggingface"
    hf_hub = hf_home / "hub"
    faster_whisper = root / "faster-whisper"
    torch_home = root / "torch"

    for path in (root, hf_home, hf_hub, faster_whisper, torch_home):
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_hub))
    os.environ.setdefault("TORCH_HOME", str(torch_home))

    data_mount = Path("/data")
    if data_mount.is_dir():
        try:
            from pipeline.storage import configure_data_root

            configure_data_root(data_mount)
        except ImportError:
            pass

    return root


def hf_hub_cache() -> Path:
    configure()
    return Path(os.environ["HUGGINGFACE_HUB_CACHE"])


def faster_whisper_root() -> Path:
    configure()
    return cache_root() / "faster-whisper"


def _repo_snapshot_cached(repo_id: str, cache_dir: Path) -> bool:
    slug = "models--" + repo_id.replace("/", "--")
    snapshots = cache_dir / slug / "snapshots"
    if not snapshots.is_dir():
        return False
    return any(p.is_dir() for p in snapshots.iterdir())


def hf_model_cached(repo_id: str) -> bool:
    """Return True if a Hugging Face repo snapshot exists in the model cache."""
    configure()
    return _repo_snapshot_cached(repo_id, hf_hub_cache())


def _whisper_hf_repo(model_size: str) -> str:
    try:
        from models.whisper import whisper_hf_repo
    except ImportError:
        from whisper import whisper_hf_repo
    return whisper_hf_repo(model_size)


def whisper_model_cached(model_size: str) -> bool:
    return _repo_snapshot_cached(_whisper_hf_repo(model_size), faster_whisper_root())
