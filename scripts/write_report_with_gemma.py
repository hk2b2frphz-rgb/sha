#!/usr/bin/env python3
"""Write the discussion section of an auto-research report with a local Gemma.

Runs inside the `gemma_runtime` uv env, invoked by autoresearch/analyst.py once
the search has finished.

  --request  JSON written by the loop (leaderboard, knob distributions, failures)
  --out      JSON written back: {"analysis": "<markdown>"}

The output is prose, not a decision: the loop pastes it into report.md under a
heading that marks it as machine-written.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemma-written run report")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.3)
    return parser.parse_args()


def knob_lines(knobs: dict[str, Any]) -> str:
    lines = []
    for name, dist in knobs.items():
        top = ", ".join(f"{v} x{c}" for v, c in (dist.get("in_top_trials") or {}).items())
        overall = ", ".join(f"{v} x{c}" for v, c in (dist.get("in_all_trials") or {}).items())
        lines.append(f"- {name}: 上位={top or '-'} / 全体={overall or '-'}")
    return "\n".join(lines)


def trial_lines(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(なし)"
    out = []
    for row in rows:
        score = row.get("score")
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "-"
        flags = " [early-stopped]" if row.get("early_stopped") else ""
        out.append(
            f"- trial {row.get('index')}: score={score_text}{flags} "
            f"({row.get('minutes', '-')}分, {row.get('source', '-')})\n"
            f"  {json.dumps(row.get('config'), ensure_ascii=False, sort_keys=True)}"
        )
    return "\n".join(out)


def build_prompt(request: dict[str, Any]) -> str:
    baseline = request.get("baseline")
    best = request.get("best")
    baseline_text = (
        f"ベースライン: trial {baseline['index']} score={baseline['score']:.4f}"
        if baseline
        else "ベースライン: 記録なし"
    )
    best_text = (
        f"最良: trial {best['index']} score={best['score']:.4f}\n"
        f"  {json.dumps(best['config'], ensure_ascii=False, sort_keys=True)}"
        if best
        else "最良: なし"
    )
    failures = request.get("failures") or []
    failure_text = (
        "\n".join(f"- trial {f['index']} ({f['status']}): {f['error']}" for f in failures)
        or "(失敗なし)"
    )
    return f"""あなたは日本語音声認識 (whisper-large-v3-turbo の LoRA fine-tune) の
ハイパーパラメータ探索の結果を読み、次の担当者に渡す考察を書く担当です。
目的関数は {request.get('objective', 'cer')} で、値が小さいほど良い。

## 実行概要
試行数: {request.get('n_trials')} (成功 {request.get('n_completed')}), GPU時間: {request.get('gpu_hours')} 時間
{baseline_text}
{best_text}

## 上位試行
{trial_lines(request.get('leaderboard') or [])}

## 下位試行
{trial_lines(request.get('worst') or [])}

## 失敗した試行
{failure_text}

## 各設定値の分布 (上位試行 / 全試行)
{knob_lines(request.get('knobs') or {})}

上記だけを根拠に、日本語の Markdown で次の 4 節を書いてください。
箇条書き中心、全体で 600 字程度。見出しは `###` を使う。

### 効いた設定
上位試行に偏って現れている値を挙げ、なぜ効いたと考えられるか。
### 効かなかった/判断できない設定
上位でも値がばらついているものは「差が出ていない」と正直に書く。
### 次に試すべきこと
具体的な値の範囲まで書く。
### 注意点
試行数が少ない、dev が小さい、早期打ち切りの影響など、
この結論を鵜呑みにできない理由があれば書く。

事実にない数値を作らないこと。データから言えないことは「判断できない」と書くこと。"""


def main() -> None:
    args = parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))

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

    messages = [{"role": "user", "content": [{"type": "text", "text": build_prompt(request)}]}]
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
        raw = raw.get("generated_text") or raw.get("text") or ""
    analysis = str(raw).strip()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"analysis": analysis, "model": args.model}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[analyst] {len(analysis)} chars -> {args.out}")


if __name__ == "__main__":
    main()
