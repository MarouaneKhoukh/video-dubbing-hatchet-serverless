#!/usr/bin/env python3
"""
Coqui TTS synthesis container.

Usage (inside container):
    python synthesize.py /data/audio/translated.txt tts_models/de/thorsten/vits /data/audio/dubbed.wav

Reads translated text, synthesizes speech, writes WAV to output path.
"""

import sys
from pathlib import Path

import torch
from TTS.api import TTS


def main():
    text_path = Path(sys.argv[1])
    model_name = sys.argv[2] if len(sys.argv) > 2 else "tts_models/de/thorsten/vits"
    output_path = Path(sys.argv[3]) if len(sys.argv) > 3 else text_path.with_suffix(".wav")

    use_gpu = torch.cuda.is_available()
    print(f"Loading TTS model: {model_name} | GPU: {use_gpu}", flush=True)

    text = text_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"No text found in {text_path}")

    print(f"Synthesizing {len(text)} characters...", flush=True)
    tts = TTS(model_name=model_name, gpu=use_gpu)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tts.tts_to_file(text=text, file_path=str(output_path))

    print(f"Dubbed audio written to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
