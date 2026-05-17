#!/usr/bin/env python3
"""
MADLAD-400 translation container.

Usage (inside container):
    python translate.py /data/audio/transcript.txt de /data/audio/translated.txt
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def split_text(text: str, max_chars: int = 700) -> list[str]:
    text = " ".join(text.split())
    chunks, cur = [], ""
    for sent in text.split(". "):
        s = sent.strip()
        if not s:
            continue
        s = s if s.endswith(".") else s + "."
        if len(cur) + len(s) + 1 > max_chars:
            if cur:
                chunks.append(cur.strip())
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        chunks.append(cur)
    return chunks


def main():
    input_path = Path(sys.argv[1])
    target_lang = sys.argv[2] if len(sys.argv) > 2 else "de"
    output_path = Path(sys.argv[3]) if len(sys.argv) > 3 else input_path.with_name("translated.txt")

    text = input_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"No text found in {input_path}")

    model_name = "jbochi/madlad400-3b-mt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )

    chunks = split_text(text)
    print(f"Translating {len(chunks)} chunks → {target_lang}", flush=True)

    translated_chunks = []
    for i, chunk in enumerate(chunks):
        prompt = f"<2{target_lang}> {chunk}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        output = model.generate(**inputs, max_new_tokens=512)
        translated = tokenizer.decode(output[0], skip_special_tokens=True)
        translated_chunks.append(translated)
        print(f"Chunk {i + 1}/{len(chunks)} done", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(" ".join(translated_chunks).strip(), encoding="utf-8")
    print(f"Translation written to {output_path}", flush=True)


if __name__ == "__main__":
    main()
