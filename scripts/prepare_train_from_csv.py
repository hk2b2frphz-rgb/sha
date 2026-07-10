#!/usr/bin/env python3
"""generated_sentences.csv -> TTS入力JSONL。

CSV列: Number, Word, Reading, Sentance, AI_Reading, Confirmed_Reading

各行について、Sentance 内の Word を Reading に置換した文を tts_text とし
(専門用語だけをひらがな読みにした発話)、学習ターゲットには元の Sentance
(漢字のまま) を残す。後段の build_whisper_manifest.py が id で join する。

出力JSONL 1行:
  {"id", "term", "reading", "sentence", "tts_text"}
    sentence : 元の Sentance (漢字) = ASR学習の正解テキスト
    tts_text : Word->Reading 置換後 = Qwen3-TTS へ渡す発話テキスト
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# CSVヘッダの表記ゆれを吸収する (Sentance は原typo、Sentence も許容)。
COLUMN_ALIASES = {
    "number": ("Number", "number", "No", "no", "id", "ID"),
    "word": ("Word", "word", "term", "Term"),
    "reading": ("Reading", "reading", "yomi", "Yomi"),
    "sentence": ("Sentance", "Sentence", "sentence", "sentance"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="generated_sentences.csv を TTS入力JSONL へ変換")
    parser.add_argument("--input", type=Path, required=True, help="generated_sentences.csv")
    parser.add_argument("--out", type=Path, required=True, help="出力JSONL")
    parser.add_argument("--id-prefix", default="gs")
    return parser.parse_args()


def resolve_column(fieldnames: list[str], key: str) -> str:
    for candidate in COLUMN_ALIASES[key]:
        if candidate in fieldnames:
            return candidate
    raise SystemExit(
        f"CSVに必要な列 ({key}) が見つかりません。"
        f" 期待する候補: {COLUMN_ALIASES[key]} / 実際: {fieldnames}"
    )


def main() -> None:
    args = parse_args()
    # utf-8-sig: Excel由来のBOM付きCSVでも先頭列名が壊れないように。
    with args.input.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise SystemExit(f"CSVヘッダを読めません: {args.input}")
        fields = list(reader.fieldnames)
        col_num = resolve_column(fields, "number")
        col_word = resolve_column(fields, "word")
        col_reading = resolve_column(fields, "reading")
        col_sentence = resolve_column(fields, "sentence")
        rows = list(reader)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    missing_term = 0
    with args.out.open("w", encoding="utf-8") as out_fh:
        for i, row in enumerate(rows, 1):
            sentence = (row.get(col_sentence) or "").strip()
            term = (row.get(col_word) or "").strip()
            reading = (row.get(col_reading) or "").strip()
            if not sentence:
                skipped += 1
                continue
            raw_num = (row.get(col_num) or "").strip()
            num = raw_num if raw_num.isdigit() else str(i)
            rec_id = f"{args.id_prefix}_{int(num):04d}"

            if term and reading:
                if term not in sentence:
                    # 表層が一致せず置換されない = TTSが漢字を誤読する恐れ。警告のみ。
                    missing_term += 1
                    print(
                        f"[WARN] {rec_id}: Word '{term}' が Sentance 内に見つからず置換されません",
                        file=sys.stderr,
                    )
                tts_text = sentence.replace(term, reading)
            else:
                tts_text = sentence

            out_fh.write(
                json.dumps(
                    {
                        "id": rec_id,
                        "term": term,
                        "reading": reading,
                        "sentence": sentence,
                        "tts_text": tts_text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

    if not written:
        raise SystemExit(f"有効な行がありません: {args.input}")
    print(
        f"wrote {written} records -> {args.out} "
        f"(skipped_empty={skipped}, term_not_found={missing_term})"
    )


if __name__ == "__main__":
    main()
