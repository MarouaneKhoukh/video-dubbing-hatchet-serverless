"""
Nested per-task pydantic settings.

Task defaults live in the config classes below. ``Settings.__init__`` loads
``.env`` when instantiated (lazy via ``get_settings()`` — not at import time).
Cloud credentials are optional until Nebius/S3/Hatchet code paths run.

Override task fields via nested env vars, e.g.:

    TRANSCRIBE__COMPUTE__PLATFORM=gpu-h200-sxm
    TRANSCRIBE__MODEL=large-v3
    TTS__LANG=e
    TARGET_LANG=es
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Nebius platform reference (verify in console):
# | GPU / CPU | platform     | example preset     |
# | L40S      | gpu-l40s-d   | 1gpu-16vcpu-96gb   |
# | H100/H200 | gpu-h100-sxm | 1gpu-16vcpu-200gb  |
# | CPU       | cpu-e2       | 4vcpu-16gb …       |


class ComputeConfig(BaseModel):
    """Nebius Serverless Job platform + preset for one pipeline task."""

    gpu: bool = False
    platform: str = "cpu-e2"
    preset: str = "4vcpu-16gb"
    preemptible: bool = False


class TaskConfig(BaseModel):
    """Shared per-task knobs. Subclasses set compute defaults and model fields."""

    image: str
    batch_size: int = 10
    max_concurrent: int = 10
    job_timeout_min: int = 60
    hatchet_timeout_s: int = 3600
    retries: int = 0
    compute: ComputeConfig = Field(default_factory=ComputeConfig)


class ExtractConfig(TaskConfig):
    image: str = "lscr.io/linuxserver/ffmpeg:latest"
    batch_size: int = 50
    max_concurrent: int = 2
    job_timeout_min: int = 20
    hatchet_timeout_s: int = 900
    compute: ComputeConfig = Field(
        default_factory=lambda: ComputeConfig(
            gpu=False, platform="cpu-e2", preset="4vcpu-16gb", preemptible=False
        )
    )


class TranscribeConfig(TaskConfig):
    image: str = "mnrozhkov/nebius-transcribe:v0.1.0"
    model: str = "distil-large-v3"
    device: str = "cuda"
    align_lang: str = "en"  # WhisperX align weights for English source audio
    batch_size: int = 10
    max_concurrent: int = 10
    job_timeout_min: int = 90
    hatchet_timeout_s: int = 4500
    retries: int = 3
    compute: ComputeConfig = Field(
        default_factory=lambda: ComputeConfig(
            gpu=True,
            platform="gpu-l40s-d",
            preset="1gpu-16vcpu-96gb",
            preemptible=True,
        )
    )


class TranslateConfig(TaskConfig):
    image: str = "mnrozhkov/nebius-translate:v0.1.0"
    model: str = "facebook/nllb-200-distilled-1.3B"
    device: str = "cuda"
    batch_size: int = 10
    max_concurrent: int = 10
    job_timeout_min: int = 40
    hatchet_timeout_s: int = 1800
    compute: ComputeConfig = Field(
        default_factory=lambda: ComputeConfig(
            gpu=False,
            platform="cpu-e2",
            preset="8vcpu-32gb",
            preemptible=False,
        )
    )


class TtsConfig(TaskConfig):
    image: str = "mnrozhkov/nebius-tts:v0.1.0"
    voice: str = "af_bella"
    lang: str = "e"  # Kokoro Spanish pipeline (EN source → ES dub)
    repo: str = "hexgrad/Kokoro-82M"
    device: str = "cuda"
    batch_size: int = 10
    max_concurrent: int = 10
    job_timeout_min: int = 60
    hatchet_timeout_s: int = 2700
    retries: int = 3
    compute: ComputeConfig = Field(
        default_factory=lambda: ComputeConfig(
            gpu=True,
            platform="gpu-l40s-d",
            preset="1gpu-16vcpu-96gb",
            preemptible=True,
        )
    )


class RemuxConfig(TaskConfig):
    image: str = "lscr.io/linuxserver/ffmpeg:latest"
    batch_size: int = 50
    max_concurrent: int = 2
    job_timeout_min: int = 20
    hatchet_timeout_s: int = 900
    compute: ComputeConfig = Field(
        default_factory=lambda: ComputeConfig(
            gpu=False, platform="cpu-e2", preset="4vcpu-16gb", preemptible=False
        )
    )


class HardwareConfig(BaseModel):
    """Shared Nebius job resources not tied to a single task."""

    job_disk_gb: int = 250
    models_bucket_prefix: str = "models"
    models_container_path: str = "/data/models"


class _EnvConfig(BaseSettings):
    """Internal loader — merges .env, process env, and code defaults."""

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        extra="ignore",
    )

    hatchet_client_token: str = ""
    nebius_iam_token: str = ""
    nebius_project_id: str = ""
    nebius_subnet_id: str = ""
    nebius_bucket_id: str = ""
    nebius_bucket_name: str = ""
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_endpoint_url: str = "https://storage.eu-north1.nebius.cloud"

    target_lang: str = "es"  # NLLB translate target (EN → ES); override via TARGET_LANG
    max_concurrent_batches: int = 5

    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    extract: ExtractConfig = Field(default_factory=ExtractConfig)
    transcribe: TranscribeConfig = Field(default_factory=TranscribeConfig)
    translate: TranslateConfig = Field(default_factory=TranslateConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    remux: RemuxConfig = Field(default_factory=RemuxConfig)


class Settings:
    """
    Pipeline settings loaded from ``.env`` on construction.

    Cloud credential fields default to empty strings so local-only runs work
    without Nebius/Hatchet configuration. Call sites that need cloud access
    should use ``require_cloud_setting()`` before using those values.
    """

    hatchet_client_token: str
    nebius_iam_token: str
    nebius_project_id: str
    nebius_subnet_id: str
    nebius_bucket_id: str
    nebius_bucket_name: str
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_endpoint_url: str
    target_lang: str
    max_concurrent_batches: int
    hardware: HardwareConfig
    extract: ExtractConfig
    transcribe: TranscribeConfig
    translate: TranslateConfig
    tts: TtsConfig
    remux: RemuxConfig

    def __init__(self, env_file: str | Path = ".env") -> None:
        path = Path(env_file)
        kwargs: dict[str, Any] = {}
        if path.is_file():
            kwargs["_env_file"] = path
        loaded = _EnvConfig(**kwargs)
        self.hatchet_client_token = loaded.hatchet_client_token
        self.nebius_iam_token = loaded.nebius_iam_token
        self.nebius_project_id = loaded.nebius_project_id
        self.nebius_subnet_id = loaded.nebius_subnet_id
        self.nebius_bucket_id = loaded.nebius_bucket_id
        self.nebius_bucket_name = loaded.nebius_bucket_name
        self.aws_access_key_id = loaded.aws_access_key_id
        self.aws_secret_access_key = loaded.aws_secret_access_key
        self.aws_endpoint_url = loaded.aws_endpoint_url
        self.target_lang = loaded.target_lang
        self.max_concurrent_batches = loaded.max_concurrent_batches
        self.hardware = loaded.hardware
        self.extract = loaded.extract
        self.transcribe = loaded.transcribe
        self.translate = loaded.translate
        self.tts = loaded.tts
        self.remux = loaded.remux


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return cached settings, loading ``.env`` on first call."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def require_cloud_setting(name: str, value: str) -> str:
    """Validate a cloud credential is configured before Nebius/S3 use."""
    if not value:
        raise RuntimeError(
            f"{name} is not set. Add it to .env (see .env.example) for cloud pipeline runs."
        )
    return value


class _SettingsProxy:
    def __getattr__(self, name: str) -> Any:
        return getattr(get_settings(), name)


settings = _SettingsProxy()
