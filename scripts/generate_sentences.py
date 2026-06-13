#!/usr/bin/env python3
"""
専門用語リストから Gemma 4 で発話例文を生成する。

入力:  1 行 1 用語のテキストファイル (--terms)
出力:  JSONL (--out)。1 行 = 1 例文:
       {"id": "0001", "term": "心筋梗塞", "sentence": "...", "tts_text": "..."}
       sentence  : 漢字を含む元の文（ASR 正解テキスト）
       tts_text  : 専門用語をひらがなに置換した文（TTS 入力用）

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

専門用語「{term}」について以下を答えてください。

1. この用語の正確な読み仮名（ひらがなのみ）
2. この用語を自然に含む話し言葉の例文を{n}個

例文の条件:
- 各例文は 1 文で、話し言葉として自然に読み上げられる長さ (15〜40 文字程度)
- 用語「{term}」を一字一句そのまま含めること
- 例文同士は場面や文型を変えて多様にすること
- 数字や記号は使わず、読み上げ可能な表現にすること

出力は以下の JSON 形式のみ。説明文は不要です。
{{"reading": "ひらがなの読み", "sentences": ["例文1", "例文2", ...]}}
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


def extract_result(raw: str, term: str) -> dict[str, Any] | None:
    """Gemma 出力から {"reading": ..., "sentences": [...]} を取り出す。"""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    reading = str(data.get("reading", "")).strip()
    sentences_raw = data.get("sentences", [])
    if not isinstance(sentences_raw, list):
        return None
    sentences = [str(s).strip() for s in sentences_raw if str(s).strip() and term in str(s)]
    if not reading or not sentences:
        return None
    return {"reading": reading, "sentences": sentences}


def generate_for_term(pipe: Any, args: argparse.Namespace, term: str) -> list[dict[str, str]]:
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
        extracted = extract_result(result_to_text(result), term)
        if extracted:
            reading = extracted["reading"]
            pairs = []
            for sentence in extracted["sentences"][: args.sentences_per_term]:
                # 置換はモデルに任せず Python で確実に行う
                tts_text = sentence.replace(term, reading)
                pairs.append({"sentence": sentence, "tts_text": tts_text})
            return pairs
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
            pairs = generate_for_term(pipe, args, term)
            if not pairs:
                failed.append(term)
                logger.error("(%d/%d) 「%s」: 生成失敗", ti, len(terms), term)
                continue
            for pair in pairs:
                count += 1
                record = {
                    "id": f"{count:04d}",
                    "term": term,
                    "sentence": pair["sentence"],
                    "tts_text": pair["tts_text"],
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                logger.info("  [%04d] %s  →  %s", count, pair["sentence"], pair["tts_text"])
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
