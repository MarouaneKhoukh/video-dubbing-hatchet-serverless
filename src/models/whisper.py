"""faster-whisper / Hugging Face repo IDs for ASR model aliases."""


def whisper_hf_repo(model_size: str) -> str:
    """HF repo ID used by faster-whisper for a given ``model_size`` alias."""
    if model_size.startswith("distil-"):
        return f"Systran/faster-distil-whisper-{model_size.removeprefix('distil-')}"
    return f"Systran/faster-whisper-{model_size}"
