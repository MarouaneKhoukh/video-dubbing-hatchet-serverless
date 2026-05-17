#!/usr/bin/env python3
"""
Whisper transcription container.

Usage (inside container):
    python transcribe.py /data/audio/episode.wav turbo cuda

Writes transcript to /data/audio/episode.txt
"""

import sys
from pathlib import Path

from faster_whisper import WhisperModel


def main():
    audio_path = Path(sys.argv[1])
    model_size = sys.argv[2] if len(sys.argv) > 2 else "turbo"
    device = sys.argv[3] if len(sys.argv) > 3 else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    output_path = audio_path.with_suffix(".txt")

    print(f"Loading Whisper model: {model_size} on {device}", flush=True)
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print(f"Transcribing: {audio_path}", flush=True)
    segments, info = model.transcribe(str(audio_path), beam_size=5)

    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})", flush=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for segment in segments:
            line = segment.text.strip()
            if line:
                f.write(line + "\n")

    print(f"Transcript written to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
