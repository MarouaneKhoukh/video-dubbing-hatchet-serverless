"""Pre-download pipeline models into the shared model cache.

Pure download functions — no Typer, no Rich, no result dataclasses. Each
``download_<stage>`` invokes the upstream library's native loader pointed at
the configured cache root; the library no-ops if the snapshot is already
cached.

Entry point ``run_downloads(stages)`` fetches the needed models for a list
of stage names. Used by ``scripts/sync_models.py``.

Model IDs come from ``pipeline.config.PipelineConfig`` (the same source the
runtime jobs use), so any model swap in ``config.py`` automatically applies here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from pipeline.config import get_config


def _default_host_cache() -> Path:
    """Repo-relative ``data/models/`` — the default L1/L2 cache when no env override."""
    return Path(__file__).resolve().parents[2] / "data" / "models"


def _import_cache():
    """Import the model_cache module, working under both repo layout and container layout."""
    try:
        import model_cache
    except ImportError:
        if str(Path(__file__).resolve().parent.parent) not in sys.path:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from models import model_cache
    return model_cache


def _ensure_cache_dir() -> Path:
    """Set MODEL_CACHE_DIR if unset, configure(), return the cache root."""
    if not os.environ.get("MODEL_CACHE_DIR"):
        os.environ["MODEL_CACHE_DIR"] = str(_default_host_cache())
    mc = _import_cache()
    mc.configure()
    return mc.cache_root()


def download_transcribe(*, model: str, device: str, align_lang: str) -> None:
    """Pre-download faster-whisper weights + WhisperX align model."""
    from faster_whisper import WhisperModel
    import whisperx

    mc = _import_cache()
    compute_type = "float16" if device == "cuda" else "int8"

    if not mc.whisper_model_cached(model):
        print(f"  downloading faster-whisper {model} …", flush=True)
        WhisperModel(
            model,
            device=device,
            compute_type=compute_type,
            download_root=str(mc.faster_whisper_root()),
            local_files_only=False,
        )
    else:
        print(f"  faster-whisper {model} already cached", flush=True)

    print(f"  ensuring WhisperX align model for lang={align_lang} …", flush=True)
    whisperx.load_align_model(
        language_code=align_lang,
        device=device,
        model_dir=str(mc.hf_hub_cache()),
    )


def download_translate(*, model: str) -> None:
    """Pre-download NLLB tokenizer + model."""
    from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

    mc = _import_cache()
    cache = str(mc.hf_hub_cache())
    local_only = mc.hf_model_cached(model)
    label = "already cached" if local_only else "downloading"
    print(f"  {label} translate model {model} …", flush=True)

    NllbTokenizer.from_pretrained(
        model, src_lang="eng_Latn", cache_dir=cache, local_files_only=local_only
    )
    AutoModelForSeq2SeqLM.from_pretrained(
        model, cache_dir=cache, local_files_only=local_only
    )


def download_tts(*, lang: str, repo: str) -> None:
    """Pre-download Kokoro TTS weights."""
    mc = _import_cache()
    from kokoro import KPipeline

    label = "already cached" if mc.hf_model_cached(repo) else "downloading"
    print(f"  {label} TTS pipeline {repo} (lang={lang}) …", flush=True)
    KPipeline(lang_code=lang)


def run_downloads(stages: list[str], *, device: str = "cpu") -> None:
    """Download the model(s) needed by each stage in *stages*.

    Stages can be any of: ``"transcribe"``, ``"translate"``, ``"tts"``.
    CPU-only stages (extract, remux) need nothing and are silently ignored if passed.
    """
    _ensure_cache_dir()
    cfg = get_config().pipeline

    for stage in stages:
        if stage == "transcribe":
            download_transcribe(
                model=cfg.transcribe.model,
                device=device,
                align_lang=cfg.transcribe.align_lang,
            )
        elif stage == "translate":
            download_translate(model=cfg.translate.model)
        elif stage == "tts":
            download_tts(lang=cfg.tts.lang, repo=cfg.tts.repo)
        # extract / remux: no-op
