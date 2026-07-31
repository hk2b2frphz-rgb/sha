#!/usr/bin/env python3
"""Fast batched Whisper evaluation with HF transformers (no CT2 / no streaming).

Used as the inner-loop evaluator of the auto-research loop: it scores a LoRA
adapter directly on top of the base model, so a trial never has to merge
weights or convert to CTranslate2. Metrics come from `eval/asr_text.py`, the
same code `evaluate_whisper_streaming.py` uses, so the numbers stay comparable
with the existing whisper-streaming reports (decoding policy still differs:
this is offline batched decoding, not the streaming policy).

Manifest lines may use either schema:
  {"id", "audio", "text"}        (train manifest, build_whisper_manifest.py)
  {"id", "wav",   "sentence"}    (test manifest, build_test_manifest.py)
An optional "term" / "terms" field enables domain-term recall.

Output:
  <out-dir>/predictions.jsonl
  <out-dir>/summary.json   {"cer", "wer", "term_recall", "score", ...}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.asr_text import (  # noqa: E402  (needs the sys.path bootstrap above)
    JapaneseTokenizer,
    error_rate,
    normalize_text,
    term_hit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="batched HF Whisper eval (CER/WER/term recall)")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--base-model", default=None, help="base model path / HF id")
    parser.add_argument(
        "--base-model-file",
        type=Path,
        default=Path("manifest.txt"),
        help="txt holding BASE_MODEL= (used when --base-model is omitted)",
    )
    parser.add_argument("--adapter", type=Path, default=None, help="LoRA adapter dir (optional)")
    parser.add_argument("--language", default="ja")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--prompt-text", default=None, help="Whisper initial prompt (domain terms etc.)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N rows (0=all)")
    parser.add_argument("--terms", type=Path, default=None, help="one term per line; adds term recall")
    parser.add_argument(
        "--objective",
        default="cer",
        choices=["cer", "wer", "cer_term"],
        help="score written to summary.json (lower is better)",
    )
    parser.add_argument("--term-weight", type=float, default=0.5, help="weight of (1-term_recall) for cer_term")
    parser.add_argument("--word-tokenizer", action="store_true", help="use fugashi for WER (default: on)")
    parser.add_argument("--char-wer", action="store_true", help="force character-level WER")
    return parser.parse_args()


def resolve_base_model(args: argparse.Namespace) -> str:
    if args.base_model:
        return args.base_model
    if not args.base_model_file.exists():
        raise SystemExit(f"--base-model is unset and {args.base_model_file} does not exist")
    for raw in args.base_model_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("BASE_MODEL=") and line.removeprefix("BASE_MODEL=").strip():
            return line.removeprefix("BASE_MODEL=").strip()
    for raw in args.base_model_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" not in line:
            return line
    raise SystemExit(f"no usable base model path in {args.base_model_file}")


def load_manifest(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        rec = json.loads(raw)
        audio = rec.get("audio") or rec.get("wav")
        if not audio:
            raise SystemExit(f"{path}:{line_no}: neither 'audio' nor 'wav' is present")
        reference = rec.get("text") or rec.get("sentence") or ""
        terms = rec.get("terms")
        if terms is None:
            terms = [rec["term"]] if rec.get("term") else []
        records.append(
            {
                "id": str(rec.get("id", line_no)),
                "audio": str(audio),
                "reference": str(reference),
                "terms": [str(t) for t in terms if str(t).strip()],
                "duration_sec": rec.get("duration_sec"),
            }
        )
        if limit and len(records) >= limit:
            break
    if not records:
        raise SystemExit(f"manifest is empty: {path}")
    return records


def load_16k_mono(path: str):
    import numpy as np
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return np.ascontiguousarray(wav, dtype="float32")


def build_model(args: argparse.Namespace, base_model: str):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(base_model, language=args.language, task=args.task)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    if device == "cpu":
        dtype = torch.float32

    model = WhisperForConditionalGeneration.from_pretrained(base_model, torch_dtype=dtype)
    if args.adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter), torch_dtype=dtype)
        model = model.merge_and_unload()
    model.generation_config.language = args.language
    model.generation_config.task = args.task
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.to(device)
    model.eval()
    return model, processor, device, dtype


def main() -> None:
    args = parse_args()
    import torch

    base_model = resolve_base_model(args)
    records = load_manifest(args.manifest, args.limit)
    extra_terms: list[str] = []
    if args.terms is not None and args.terms.exists():
        extra_terms = [
            line.split("\t")[0].strip()
            for line in args.terms.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]

    print(f"[eval] base={base_model} adapter={args.adapter} rows={len(records)}")
    model, processor, device, dtype = build_model(args, base_model)

    prompt_ids = None
    if args.prompt_text:
        prompt_ids = processor.get_prompt_ids(args.prompt_text, return_tensors="pt").to(device)

    tokenizer = JapaneseTokenizer(enabled=not args.char_wer)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.out_dir / "predictions.jsonl"

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "language": args.language,
        "task": args.task,
    }
    if args.beam_size > 1:
        gen_kwargs["num_beams"] = args.beam_size
    if args.temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = args.temperature
    if args.no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size
    if prompt_ids is not None:
        gen_kwargs["prompt_ids"] = prompt_ids
        gen_kwargs["prompt_condition_type"] = "first-segment"

    rows: list[dict[str, Any]] = []
    total_audio = 0.0
    started = time.monotonic()
    with predictions_path.open("w", encoding="utf-8") as out_fh:
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            waves = [load_16k_mono(rec["audio"]) for rec in batch]
            total_audio += sum(len(w) / 16000.0 for w in waves)
            features = processor.feature_extractor(
                waves, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device=device, dtype=dtype)
            with torch.no_grad():
                generated = model.generate(features, **gen_kwargs)
            texts = processor.batch_decode(generated, skip_special_tokens=True)
            for rec, hypothesis in zip(batch, texts):
                hypothesis = hypothesis.strip()
                reference = rec["reference"]
                ref_chars = list(normalize_text(reference))
                hyp_chars = list(normalize_text(hypothesis))
                terms = rec["terms"] or [t for t in extra_terms if term_hit(t, reference)]
                hits = [t for t in terms if term_hit(t, hypothesis)]
                row = {
                    "id": rec["id"],
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "cer": error_rate(ref_chars, hyp_chars),
                    "wer": error_rate(tokenizer.words(reference), tokenizer.words(hypothesis)),
                    "terms": terms,
                    "term_hits": len(hits),
                    "term_total": len(terms),
                }
                rows.append(row)
                out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            done = min(start + args.batch_size, len(records))
            print(f"[eval] {done}/{len(records)} rows", flush=True)

    wall = time.monotonic() - started
    n = len(rows)
    term_total = sum(r["term_total"] for r in rows)
    term_hits = sum(r["term_hits"] for r in rows)
    term_recall = (term_hits / term_total) if term_total else None
    cer = sum(r["cer"] for r in rows) / n
    wer = sum(r["wer"] for r in rows) / n
    if args.objective == "wer":
        score = wer
    elif args.objective == "cer_term":
        score = cer + args.term_weight * (1.0 - (term_recall if term_recall is not None else 1.0))
    else:
        score = cer

    summary = {
        "n": n,
        "base_model": base_model,
        "adapter": str(args.adapter) if args.adapter else None,
        "manifest": str(args.manifest),
        "cer": cer,
        "wer": wer,
        "tokenizer": tokenizer.mode,
        "term_recall": term_recall,
        "term_total": term_total,
        "objective": args.objective,
        "score": score,
        "beam_size": args.beam_size,
        "prompt_text": args.prompt_text,
        "total_audio_sec": total_audio,
        "total_wall_sec": wall,
        "rtf": (wall / total_audio) if total_audio else None,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
