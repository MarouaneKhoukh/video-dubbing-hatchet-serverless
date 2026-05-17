from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Hatchet
    hatchet_client_token: str

    # Nebius auth
    nebius_iam_token: str
    nebius_project_id: str
    nebius_subnet_id: str

    # Object storage
    nebius_bucket_id: str
    nebius_bucket_name: str
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_endpoint_url: str = "https://storage.eu-north1.nebius.cloud"

    # Pipeline
    target_lang: str = "de"
    whisper_image: str = "ghcr.io/your-org/nebius-whisper:latest"
    tts_image: str = "ghcr.io/your-org/nebius-tts:latest"
    translate_image: str = "ghcr.io/your-org/nebius-translate:latest"
    whisper_model: str = "turbo"
    tts_model: str = "tts_models/de/thorsten/vits"

    # Nebius job config
    gpu_platform: str = "gpu-l40s-d"
    gpu_preset: str = "1gpu-16vcpu-96gb"
    cpu_platform: str = "cpu-e2"
    cpu_preset: str = "32vcpu-128gb"
    job_disk_gb: int = 250
    job_timeout_minutes: int = 30


settings = Settings()
