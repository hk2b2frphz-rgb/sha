#!/usr/bin/env python3
"""Merge a Whisper LoRA adapter into its base model and (optionally) convert to CT2.

The auto-research loop keeps trials cheap by never merging weights; this script
turns the winning adapter into the artifacts the existing evaluation pipeline
expects (`merged_hf/` for transformers, `ct2/` for whisper-streaming).

  uv run python scripts/export_whisper_model.py \
      --adapter experiments/autoresearch/trials/trial_0007/adapter \
      --merge-dir out/whisper_turbo/merged_hf --ct2-dir out/whisper_turbo/ct2
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="merge LoRA adapter and convert to CTranslate2")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--merge-dir", type=Path, required=True)
    parser.add_argument("--ct2-dir", type=Path, default=None, help="omit to skip CT2 conversion")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--base-model-file", type=Path, default=Path("manifest.txt"))
    parser.add_argument("--language", default="ja")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--quantization", default="float16")
    return parser.parse_args()


def resolve_base_model(args: argparse.Namespace) -> str:
    if args.base_model:
        return args.base_model
    if not args.base_model_file.exists():
        raise SystemExit(f"--base-model is unset and {args.base_model_file} does not exist")
    lines = args.base_model_file.read_text(encoding="utf-8").splitlines()
    for raw in lines:
        line = raw.strip()
        if line.startswith("BASE_MODEL=") and line.removeprefix("BASE_MODEL=").strip():
            return line.removeprefix("BASE_MODEL=").strip()
    for raw in lines:
        line = raw.strip()
        if line and not line.startswith("#") and "=" not in line:
            return line
    raise SystemExit(f"no usable base model path in {args.base_model_file}")


def main() -> None:
    args = parse_args()
    if not args.adapter.exists():
        raise SystemExit(f"adapter not found: {args.adapter}")

    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    base_model = resolve_base_model(args)
    print(f"[export] base={base_model} adapter={args.adapter}")
    processor = WhisperProcessor.from_pretrained(base_model, language=args.language, task=args.task)
    model = WhisperForConditionalGeneration.from_pretrained(base_model)
    model.generation_config.language = args.language
    model.generation_config.task = args.task
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    merged = PeftModel.from_pretrained(model, str(args.adapter)).merge_and_unload()

    args.merge_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.merge_dir)
    processor.save_pretrained(args.merge_dir)
    print(f"[export] merged model -> {args.merge_dir}")

    if args.ct2_dir is None:
        return
    if args.ct2_dir.exists():
        shutil.rmtree(args.ct2_dir)
    cmd = [
        "ct2-transformers-converter",
        "--model", str(args.merge_dir),
        "--output_dir", str(args.ct2_dir),
        "--quantization", args.quantization,
        "--copy_files", "tokenizer.json", "preprocessor_config.json",
    ]
    print(f"[export] $ {' '.join(cmd)}")
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        print("[export] CT2 conversion failed; merged model is still usable", file=sys.stderr)
        raise SystemExit(completed.returncode)
    print(f"[export] ct2 model -> {args.ct2_dir}")


if __name__ == "__main__":
    main()
