#!/usr/bin/env python3
"""
PDF 文書群から専門用語を抽出する。

入力:  PDF ファイルまたは PDF を含むディレクトリ (--pdfs, 複数指定可)
出力:  1 行 1 用語のテキストファイル (--out)。generate_sentences.py にそのまま渡せる。

処理:
  1. pypdf で各 PDF をテキスト化
  2. テキストをチャンクに分割し、Gemma 4 に専門用語を JSON 配列で抽出させる
  3. 全チャンクの結果をマージして重複排除し、出現回数の多い順に出力

使い方:
  uv run --project gemma_runtime python scripts/extract_terms.py \
      --pdfs docs/ --out out/terms.txt
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("extract_terms")

PROMPT_TEMPLATE = """\
あなたは音声認識システムの評価用語彙を作成する専門家です。

以下の文章から、その分野に固有の「専門用語」を抽出してください。

条件:
- 専門用語のみ。一般的な日常語 (「会議」「報告」など) は含めない
- 名詞または複合名詞のまま抽出する (活用させない)
- 文章中に実際に出てきた表記をそのまま使う
- 最大 {max_terms} 個まで。重要なものから順に

出力は JSON 配列のみ。説明文は不要です。
例: ["心筋梗塞", "経皮的冠動脈形成術"]

--- 文章 ---
{text}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PDF から専門用語を抽出する")
    parser.add_argument(
        "--pdfs", type=Path, nargs="+", required=True,
        help="PDF ファイルまたは PDF を含むディレクトリ (複数指定可)",
    )
    parser.add_argument("--out", type=Path, required=True, help="出力テキストパス (1 行 1 用語)")
    parser.add_argument("--chunk-chars", type=int, default=3000, help="Gemma に渡すチャンクの文字数")
    parser.add_argument("--max-terms-per-chunk", type=int, default=20)
    parser.add_argument("--min-count", type=int, default=1,
                        help="この回数以上のチャンクに出現した用語のみ採用")
    parser.add_argument("--max-terms", type=int, default=0,
                        help="出力する用語数の上限 (0 = 無制限)")
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="抽出タスクなので低めが既定")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# PDF → テキスト
# ---------------------------------------------------------------------------

def collect_pdf_paths(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for p in inputs:
        if p.is_dir():
            paths.extend(sorted(p.rglob("*.pdf")))
        elif p.suffix.lower() == ".pdf":
            paths.append(p)
        else:
            logger.warning("PDF ではないためスキップ: %s", p)
    if not paths:
        raise SystemExit("PDF が見つかりません")
    return paths


def pdf_to_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:
            logger.warning("%s: ページ抽出失敗 (%s)", path.name, exc)
    text = "\n".join(pages)
    # PDF 抽出特有の崩れを軽く整える
    text = re.sub(r"[ \t　]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_chunks(text: str, chunk_chars: int) -> list[str]:
    chunks = []
    for start in range(0, len(text), chunk_chars):
        chunk = text[start:start + chunk_chars].strip()
        if len(chunk) >= 100:  # 短すぎる端切れは無視
            chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# Gemma 抽出
# ---------------------------------------------------------------------------

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


def extract_json_terms(raw: str) -> list[str]:
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    terms = []
    for item in items:
        term = str(item).strip()
        # 長すぎる/短すぎる候補はノイズとして除外
        if 2 <= len(term) <= 30:
            terms.append(term)
    return terms


def extract_from_chunk(pipe: Any, args: argparse.Namespace, chunk: str) -> list[str]:
    prompt = PROMPT_TEMPLATE.format(text=chunk, max_terms=args.max_terms_per_chunk)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": True,
        "return_full_text": False,
    }
    try:
        result = pipe(messages, **gen_kwargs)
    except TypeError:
        result = pipe(prompt, **gen_kwargs)
    return extract_json_terms(result_to_text(result))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    pdf_paths = collect_pdf_paths(args.pdfs)
    logger.info("%d 個の PDF を処理します", len(pdf_paths))

    # 1. テキスト化
    all_chunks: list[tuple[str, str]] = []  # (pdf名, チャンク)
    for path in pdf_paths:
        text = pdf_to_text(path)
        chunks = split_chunks(text, args.chunk_chars)
        logger.info("%s: %d 文字 -> %d チャンク", path.name, len(text), len(chunks))
        all_chunks.extend((path.name, c) for c in chunks)

    if not all_chunks:
        raise SystemExit("テキストを抽出できませんでした (スキャン PDF の場合は OCR が必要です)")

    # 2. Gemma で抽出
    start = time.monotonic()
    pipe = build_pipeline(args)
    logger.info("モデルロード完了 (%.1f 秒)", time.monotonic() - start)

    counter: Counter[str] = Counter()
    gen_start = time.monotonic()
    total = len(all_chunks)
    for i, (name, chunk) in enumerate(all_chunks, 1):
        terms = extract_from_chunk(pipe, args, chunk)
        counter.update(set(terms))  # 同一チャンク内の重複は 1 回と数える
        elapsed = time.monotonic() - gen_start
        eta = elapsed / i * (total - i)
        logger.info(
            "(%d/%d) %s: %d 語抽出 (累計ユニーク %d 語) | 経過 %.0f 秒 / 残り目安 %.0f 秒",
            i, total, name, len(terms), len(counter), elapsed, eta,
        )
        if terms:
            logger.info("  %s", ", ".join(terms[:10]) + (" ..." if len(terms) > 10 else ""))

    # 3. 集計して出力
    selected = [(t, c) for t, c in counter.most_common() if c >= args.min_count]
    if args.max_terms > 0:
        selected = selected[: args.max_terms]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        fh.write(f"# {len(pdf_paths)} PDF / {total} チャンクから抽出 (出現チャンク数順)\n")
        for term, count in selected:
            fh.write(f"{term}\n")

    logger.info("完了: %d 語を %s に保存", len(selected), args.out)
    if not selected:
        sys.exit(1)


if __name__ == "__main__":
    main()
