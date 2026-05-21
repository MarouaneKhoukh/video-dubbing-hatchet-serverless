#!/usr/bin/env python3
"""
NLLB-200 translation job.

Invocation modes:

  Single-file (legacy):
      python3 /translate.py <input_txt> <target_lang> <output_txt> [model_name]

  Run manifest batch:
      python3 /translate.py /data/runs/<run_id>/manifests/translate.json [chunk_idx]
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

try:
    import model_cache
except ImportError:
    from models import model_cache

from pipeline.metadata import (
    config_str,
    ensure_torch_device,
    load_manifest,
    parse_manifest_argv,
    parse_task_runtime,
    resolve_chunk,
)

model_cache.configure()
DEFAULTS = model_cache.default_model_spec()

DATA = Path("/data")

_NLLB_LANG: dict[str, str] = {
    "de": "deu_Latn", "fr": "fra_Latn", "es": "spa_Latn",
    "it": "ita_Latn", "pt": "por_Latn", "nl": "nld_Latn",
    "pl": "pol_Latn", "ru": "rus_Cyrl", "zh": "zho_Hans",
    "ja": "jpn_Jpan", "ko": "kor_Hang", "ar": "arb_Arab",
    "hi": "hin_Deva", "tr": "tur_Latn",
}


def _split_text(text: str, max_chars: int = 700) -> list[str]:
    text = " ".join(text.split())
    chunks: list[str] = []
    cur = ""
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


def _translate_one(
    tokenizer: NllbTokenizer,
    model: AutoModelForSeq2SeqLM,
    forced_bos_token_id: int,
    input_path: Path,
    output_path: Path,
    *,
    force: bool = False,
) -> None:
    if not force and output_path.exists():
        print(f"SKIP (already done): {input_path.name}", flush=True)
        return
    print(f"FILE: {input_path.name} -> {output_path.name}", flush=True)
    text = input_path.read_text(encoding="utf-8")
    chunks = _split_text(text)
    print(f"  chunks: {len(chunks)}", flush=True)

    translated: list[str] = []
    for i, chunk in enumerate(chunks):
        inputs = tokenizer(
            chunk, return_tensors="pt", truncation=True, max_length=1024
        ).to(model.device)
        out = model.generate(
            **inputs, forced_bos_token_id=forced_bos_token_id, max_new_tokens=512
        )
        translated.append(tokenizer.decode(out[0], skip_special_tokens=True))
        print(f"  chunk {i + 1}/{len(chunks)} done", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(" ".join(translated).strip(), encoding="utf-8")


def _load_model(
    model_name: str,
    target_lang: str,
    device: str,
) -> tuple[NllbTokenizer, AutoModelForSeq2SeqLM, int]:
    ensure_torch_device(device)
    tgt = _NLLB_LANG.get(target_lang, target_lang)
    cache = str(model_cache.hf_hub_cache())
    cached = model_cache.hf_model_cached(model_name)
    if cached:
        print(f"Using cached {model_name}", flush=True)
    else:
        print(f"Downloading {model_name} → {cache}", flush=True)
    print(f"Loading {model_name} on {device} (target={target_lang} → {tgt})", flush=True)
    tokenizer = NllbTokenizer.from_pretrained(
        model_name,
        src_lang="eng_Latn",
        cache_dir=cache,
        local_files_only=cached,
    )
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        cache_dir=cache,
        local_files_only=cached,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt)
    return tokenizer, model, forced_bos_token_id


def _run_manifest(manifest_path: Path, chunk_idx: int) -> None:
    manifest = load_manifest(manifest_path)
    runtime = parse_task_runtime(manifest, "translate")
    cfg = runtime["config"]
    target_lang = runtime["target_lang"]
    model_name = config_str(cfg, "model")
    device = ensure_torch_device(config_str(cfg, "device"))
    force = runtime["force"]
    files = resolve_chunk(manifest, chunk_idx)

    print(
        f"MANIFEST: {manifest_path.name} chunk={chunk_idx} | {len(files)} files | "
        f"model={model_name} device={device} force={force}",
        flush=True,
    )
    tokenizer, model, forced = _load_model(model_name, target_lang, device)

    for idx, item in enumerate(files, 1):
        print(f"\n[{idx}/{len(files)}]", flush=True)
        _translate_one(
            tokenizer, model, forced,
            DATA / item["transcript_key"],
            DATA / item["translated_key"],
            force=force,
        )

    print(f"\nChunk complete: {len(files)} files processed", flush=True)


def _run_single(
    input_path: Path,
    target_lang: str,
    output_path: Path,
    model_name: str,
    *,
    device: str = "cpu",
) -> None:
    tokenizer, model, forced = _load_model(model_name, target_lang, device)
    _translate_one(tokenizer, model, forced, input_path, output_path)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: translate.py <input_txt> <target_lang> <output_txt> [model]  OR  "
            "translate.py <task_manifest.json> [chunk_idx]"
        )
    arg1 = Path(sys.argv[1])
    if arg1.suffix == ".json":
        manifest_path, chunk_idx = parse_manifest_argv()
        _run_manifest(manifest_path, chunk_idx)
        return
    target_lang = sys.argv[2]
    output_path = Path(sys.argv[3])
    model_name = sys.argv[4] if len(sys.argv) > 4 else DEFAULTS.translate_model
    _run_single(arg1, target_lang, output_path, model_name)


if __name__ == "__main__":
    main()
