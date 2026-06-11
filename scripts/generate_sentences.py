#!/usr/bin/env python3
"""
専門用語リストから Gemma 4 で発話例文を生成する。

入力:  1 行 1 用語のテキストファイル (--terms)
出力:  JSONL (--out)。1 行 = 1 例文:
       {"id": "0001", "term": "心筋梗塞", "sentence": "..."}

使い方:
  uv run --project gemma_runtime python scripts/generate_sentences.py \
      --terms terms_example.txt --out out/sentences.jsonl \
      --sentences-per-term 3
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("generate_sentences")

PROMPT_TEMPLATE = """\
あなたは音声認識システムのテスト用例文を作成するアシスタントです。

専門用語「{term}」を必ず含む、自然な日本語の話し言葉の例文を{n}個作成してください。

条件:
- 各例文は 1 文で、話し言葉として自然に読み上げられる長さ (15〜40 文字程度)
- 用語「{term}」を一字一句そのまま含めること
- 例文同士は場面や文型を変えて多様にすること
- 数字や記号は使わず、読み上げ可能な表現にすること

出力は JSON 配列のみ。説明文は不要です。
例: ["例文1", "例文2", "例文3"]
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="専門用語から発話例文を生成する")
    parser.add_argument("--terms", type=Path, required=True, help="1 行 1 用語のテキストファイル")
    parser.add_argument("--out", type=Path, required=True, help="出力 JSONL パス")
    parser.add_argument("--sentences-per-term", type=int, default=3)
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-retries", type=int, default=2, help="JSON パース失敗時の再生成回数")
    return parser.parse_args()


def load_terms(path: Path) -> list[str]:
    terms = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    terms = [t for t in terms if t and not t.startswith("#")]
    if not terms:
        raise SystemExit(f"用語が見つかりません: {path}")
    return terms


def build_pipeline(args: argparse.Namespace) -> Any:
    import torch
    from transformers import pipeline

    kwargs: dict[str, Any] = {
        "model": args.model,
        "device_map": args.device_map,
        "model_kwargs": {"attn_implementation": "sdpa"},
    }
    if args.dtype != "auto":
        kwargs["torch_dtype"] = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[args.dtype]

    logger.info("Gemma をロード中: %s", args.model)
    try:
        return pipeline("text-generation", **kwargs)
    except (TypeError, ValueError):
        kwargs.pop("model_kwargs", None)
        return pipeline("text-generation", **kwargs)


def result_to_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        return "\n".join(result_to_text(item) for item in result)
    if isinstance(result, dict):
        for key in ("generated_text", "text", "content"):
            if key in result:
                return result_to_text(result[key])
        return json.dumps(result, ensure_ascii=False)
    return str(result)


def extract_sentences(raw: str, term: str) -> list[str]:
    """Gemma 出力から JSON 配列を取り出し、用語を含む文だけ残す。"""
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    sentences = [str(s).strip() for s in items if isinstance(s, str) and str(s).strip()]
    kept = [s for s in sentences if term in s]
    dropped = len(sentences) - len(kept)
    if dropped:
        logger.warning("用語「%s」を含まない例文を %d 件除外", term, dropped)
    return kept


def generate_for_term(pipe: Any, args: argparse.Namespace, term: str) -> list[str]:
    prompt = PROMPT_TEMPLATE.format(term=term, n=args.sentences_per_term)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": True,
        "return_full_text": False,
    }
    for attempt in range(1 + args.max_retries):
        try:
            result = pipe(messages, **gen_kwargs)
        except TypeError:
            result = pipe(prompt, **gen_kwargs)
        sentences = extract_sentences(result_to_text(result), term)
        if sentences:
            return sentences[: args.sentences_per_term]
        logger.warning("「%s」: 生成結果のパースに失敗 (attempt %d)", term, attempt + 1)
    return []


def main() -> None:
    args = parse_args()
    terms = load_terms(args.terms)
    logger.info("%d 用語 x %d 文を生成します", len(terms), args.sentences_per_term)

    start = time.monotonic()
    pipe = build_pipeline(args)
    logger.info("モデルロード完了 (%.1f 秒)", time.monotonic() - start)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    failed: list[str] = []
    gen_start = time.monotonic()
    with args.out.open("w", encoding="utf-8") as fh:
        for ti, term in enumerate(terms, 1):
            logger.info("--- (%d/%d) 用語「%s」を生成中 ---", ti, len(terms), term)
            sentences = generate_for_term(pipe, args, term)
            if not sentences:
                failed.append(term)
                logger.error("(%d/%d) 「%s」: 生成失敗", ti, len(terms), term)
                continue
            for sentence in sentences:
                count += 1
                record = {"id": f"{count:04d}", "term": term, "sentence": sentence}
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                logger.info("  [%04d] %s", count, sentence)
            fh.flush()  # 中断してもここまでの結果は out に残る
            elapsed = time.monotonic() - gen_start
            eta = elapsed / ti * (len(terms) - ti)
            logger.info(
                "(%d/%d) 完了: 累計 %d 文 / 経過 %.0f 秒 / 残り目安 %.0f 秒",
                ti, len(terms), count, elapsed, eta,
            )

    logger.info("完了: %d 文を %s に保存", count, args.out)
    if failed:
        logger.error("生成に失敗した用語: %s", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
