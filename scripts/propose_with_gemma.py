#!/usr/bin/env python3
"""Ask a local Gemma for the next Whisper fine-tuning configs to try.

Runs inside the `gemma_runtime` uv env (`uv run --project gemma_runtime`),
invoked as a subprocess by autoresearch/proposer.py so an LLM failure can never
take the search loop down with it.

  --request  JSON written by the loop: {objective, space, history, best, ...}
  --out      JSON written back:        {"proposals": [ {...}, ... ], "raw": "..."}

Exits non-zero only on unrecoverable errors; the loop falls back to its
evolutionary proposer whenever no usable proposal comes back.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemma-based config proposer")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    return parser.parse_args()


def extract_json_text(text: str) -> str:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return stripped
    starts = [i for i in (stripped.find("{"), stripped.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON object or array in model output")
    start = min(starts)
    end = max(stripped.rfind("}"), stripped.rfind("]"))
    if end < start:
        raise ValueError("unterminated JSON in model output")
    return stripped[start : end + 1]


def parse_proposals(raw: str) -> list[dict[str, Any]]:
    parsed = json.loads(extract_json_text(raw))
    rows = parsed.get("proposals", parsed) if isinstance(parsed, dict) else parsed
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list):
        raise ValueError("proposals must be a list of objects")
    proposals: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            config = row.get("config") if isinstance(row.get("config"), dict) else row
            proposals.append({k: v for k, v in config.items() if k != "reason"})
    return proposals


def history_table(history: list[dict[str, Any]]) -> str:
    if not history:
        return "(no trials yet)"
    lines = []
    for row in history:
        score = row.get("score")
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else f"FAILED({row.get('status')})"
        metrics = row.get("metrics") or {}
        extra = " ".join(
            f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))
        )
        # Screening trials train for a few minutes only, so their scores must not
        # be compared with full-length ones; the stage tag makes that visible.
        stage = row.get("stage") or "full"
        lines.append(
            f"trial {row.get('index')} [{stage}]: score={score_text} {extra}\n"
            f"  config={json.dumps(row.get('config'), ensure_ascii=False, sort_keys=True)}"
        )
    return "\n".join(lines)


def build_prompt(request: dict[str, Any]) -> str:
    best = request.get("best")
    best_text = (
        f"Best so far: trial {best['index']} score={best['score']:.4f}\n"
        f"  config={json.dumps(best['config'], ensure_ascii=False, sort_keys=True)}"
        if best
        else "Best so far: none yet."
    )
    n = int(request.get("n_proposals", 3))
    guidance = (request.get("guidance") or "").strip()
    guidance_block = (
        f"\nNotes from the human running this search (treat as authoritative):\n{guidance}\n"
        if guidance
        else ""
    )
    return f"""You are tuning a LoRA fine-tune of whisper-large-v3-turbo for Japanese
speech that contains domain-specific technical terms. The objective is to
{request.get('objective', 'minimize dev CER')} (LOWER score is better).

Search space (stay strictly inside these domains):
{request.get('space', '')}
{guidance_block}

{best_text}

Previous trials (most recent last):
{history_table(request.get('history') or [])}

Propose the {n} most promising NEXT configurations.
Rules:
- Every proposal must set every key in the search space.
- Do not repeat a configuration that already appears above.
- Vary a small number of keys at a time from a strong configuration, but include
  at least one clearly exploratory proposal.
- Reason about which knobs the history suggests actually matter.

Reply with JSON only, no prose, in exactly this shape:
{{"proposals": [{{"reason": "<short>", "config": {{...}}}}]}}"""


def main() -> None:
    args = parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    prompt = build_prompt(request)

    import torch
    from transformers import pipeline

    kwargs: dict[str, Any] = {"model": args.model, "device_map": args.device_map}
    if args.dtype != "auto":
        kwargs["torch_dtype"] = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[args.dtype]
    try:
        pipe = pipeline("text-generation", model_kwargs={"attn_implementation": "sdpa"}, **kwargs)
    except (TypeError, ValueError):
        pipe = pipeline("text-generation", **kwargs)

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    outputs = pipe(
        messages,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature if args.temperature > 0 else None,
        return_full_text=False,
    )
    raw = outputs
    while isinstance(raw, list) and raw:
        raw = raw[0]
    if isinstance(raw, dict):
        raw = raw.get("generated_text") or raw.get("text") or json.dumps(raw, ensure_ascii=False)
    raw = str(raw)

    try:
        proposals = parse_proposals(raw)
        error = None
    except (ValueError, json.JSONDecodeError) as exc:
        proposals, error = [], str(exc)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"proposals": proposals, "raw": raw[-4000:], "error": error, "model": args.model},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[proposer] {len(proposals)} proposal(s) -> {args.out}")
    if error:
        print(f"[proposer] parse error: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
