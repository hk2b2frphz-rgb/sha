#!/usr/bin/env python3
"""
専門用語の読み仮名を GPT で確認し、TTS 用テキストを生成する (2026-06-13)。

generate_sentences.py の出力 JSONL を受け取り、専門用語をひらがなに置換した
tts_text フィールドを追加して出力する。synthesize_speech.py に渡す前に実行する。

使い方:
  export OPENAI_API_KEY=sk-...
  uv run python scripts/annotate_readings.py \
      --sentences out/sentences.jsonl \
      --out out/sentences_annotated.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("annotate_readings")

SYSTEM_PROMPT = """\
あなたは日本語の専門用語の読み仮名を確認するアシスタントです。
与えられた文と専門用語について、その用語の正確な読み仮名（ひらがな）を答え、
文中のその用語をひらがなに置き換えた文を返します。
出力は JSON のみ。説明文は不要です。
"""

USER_PROMPT_TEMPLATE = """\
専門用語「{term}」を含む以下の文について、用語の読み仮名を確認してください。

文: {sentence}

以下の JSON 形式で返してください（他のテキストは不要）:
{{"reading": "ひらがなの読み", "tts_text": "文中の専門用語をひらがなに置き換えた文"}}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="専門用語の読み仮名を LLM API で確認し tts_text を付与する")
    parser.add_argument("--sentences", type=Path, required=True, help="generate_sentences.py の出力 JSONL")
    parser.add_argument("--out", type=Path, required=True, help="出力 JSONL パス")
    parser.add_argument("--provider", default="openai", choices=["openai", "gemini"],
                        help="使用する API プロバイダ (default: openai)")
    parser.add_argument("--model", default=None, help="モデル名（省略時はプロバイダのデフォルト）")
    parser.add_argument("--base-url", default=None, help="OpenAI 互換 API の base URL（カスタム用）")
    parser.add_argument("--max-retries", type=int, default=2, help="JSON パース失敗時の再試行回数")
    parser.add_argument("--interval", type=float, default=0.5, help="API 呼び出し間隔（秒）")
    return parser.parse_args()


PROVIDER_DEFAULTS = {
    "openai": {
        "base_url": None,
        "env_key": "OPENAI_API_KEY",
        "model": "gpt-4o",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "env_key": "GEMINI_API_KEY",
        "model": "gemini-2.0-flash",
    },
}


def get_client(args: argparse.Namespace) -> Any:
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai パッケージが見つかりません。`uv add openai` でインストールしてください。")
        sys.exit(1)
    defaults = PROVIDER_DEFAULTS.get(args.provider, PROVIDER_DEFAULTS["openai"])
    api_key = os.environ.get(defaults["env_key"])
    if not api_key:
        logger.error("%s 環境変数が設定されていません。", defaults["env_key"])
        sys.exit(1)
    base_url = args.base_url or defaults["base_url"]
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def annotate(client: Any, model: str, term: str, sentence: str, max_retries: int) -> dict[str, str]:
    """GPT に読み仮名を確認させ {"reading": ..., "tts_text": ...} を返す。失敗時は元文を返す。"""
    prompt = USER_PROMPT_TEMPLATE.format(term=term, sentence=sentence)
    for attempt in range(1 + max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or ""
            data = json.loads(raw)
            reading = str(data.get("reading", "")).strip()
            tts_text = str(data.get("tts_text", "")).strip()
            if reading and tts_text:
                return {"reading": reading, "tts_text": tts_text}
            logger.warning("「%s」: 不完全なレスポンス (attempt %d): %s", term, attempt + 1, raw[:100])
        except (json.JSONDecodeError, KeyError, Exception) as e:
            logger.warning("「%s」: エラー (attempt %d): %s", term, attempt + 1, e)
    logger.error("「%s」: 読み仮名の取得に失敗。元の文をそのまま使います。", term)
    return {"reading": "", "tts_text": sentence}


def load_sentences(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    if not records:
        raise SystemExit(f"例文が見つかりません: {path}")
    return records


def main() -> None:
    args = parse_args()
    records = load_sentences(args.sentences)
    logger.info("%d 文の読み仮名を確認します (model=%s)", len(records), args.model)

    if args.model is None:
        args.model = PROVIDER_DEFAULTS.get(args.provider, PROVIDER_DEFAULTS["openai"])["model"]
    logger.info("プロバイダ: %s  モデル: %s", args.provider, args.model)
    client = get_client(args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    start = time.monotonic()

    with args.out.open("w", encoding="utf-8") as fh:
        for i, rec in enumerate(records, 1):
            term = rec.get("term", "")
            sentence = rec["sentence"]
            result = annotate(client, args.model, term, sentence, args.max_retries)

            if not result["reading"]:
                failed.append(f"{rec['id']}:{term}")

            out_rec = {**rec, "reading": result["reading"], "tts_text": result["tts_text"]}
            fh.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            fh.flush()

            elapsed = time.monotonic() - start
            eta = elapsed / i * (len(records) - i)
            logger.info(
                "(%d/%d) 「%s」→ 読み: %s | 経過 %.0f 秒 / 残り目安 %.0f 秒",
                i, len(records), term, result["reading"], elapsed, eta,
            )
            if i < len(records):
                time.sleep(args.interval)

    logger.info("完了: %s に %d 件を保存", args.out, len(records))
    if failed:
        logger.warning("読み取得に失敗した用語 (元の文を使用): %s", ", ".join(failed))


if __name__ == "__main__":
    main()
