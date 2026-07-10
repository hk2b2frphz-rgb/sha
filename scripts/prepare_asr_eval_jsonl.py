#!/usr/bin/env python3
"""Plain text examples -> JSONL records for Qwen3-TTS evaluation data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="例文txtをsynthesize_speech.py用JSONLへ変換する")
    parser.add_argument("--input", type=Path, required=True, help="1行1例文のtxt、TSV、またはJSONL")
    parser.add_argument("--out", type=Path, required=True, help="出力JSONL")
    parser.add_argument("--id-prefix", default="eval")
    parser.add_argument("--text-column", type=int, default=0, help="TSV入力時の例文列(0始まり)")
    parser.add_argument("--term-column", type=int, default=-1, help="TSV入力時の用語列(0始まり、-1で空)")
    parser.add_argument(
        "--readings-tsv",
        type=Path,
        default=None,
        help="term, reading の2列TSV。sentence内のtermをtts_textでreadingへ置換する",
    )
    return parser.parse_args()


def load_readings(path: Path | None) -> list[tuple[str, str]]:
    if path is None:
        return []
    rows = path.read_text(encoding="utf-8").splitlines()
    if not rows:
        raise SystemExit(f"読みTSVが空です: {path}")
    first_cols = rows[0].rstrip("\n").split("\t")
    has_header = len(first_cols) >= 2 and first_cols[0] == "term" and first_cols[1] == "reading"
    start = 1 if has_header else 0
    readings: list[tuple[str, str]] = []
    for line_no, raw in enumerate(rows[start:], start + 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cols = raw.split("\t")
        if len(cols) < 2:
            raise SystemExit(f"{path}:{line_no}: term と reading の2列が必要です")
        term = cols[0].strip()
        reading = cols[1].strip()
        if term and reading:
            readings.append((term, reading))
    if not readings:
        raise SystemExit(f"有効なterm/readingが見つかりません: {path}")
    return sorted(readings, key=lambda item: len(item[0]), reverse=True)


def apply_readings(text: str, readings: list[tuple[str, str]]) -> str:
    for term, reading in readings:
        text = text.replace(term, reading)
    return text


def iter_records(path: Path, text_column: int, term_column: int, readings: list[tuple[str, str]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            rec = json.loads(line)
            text = str(rec.get("tts_text") or rec.get("sentence") or rec.get("text") or "").strip()
            if not text:
                raise SystemExit(f"{path}:{line_no}: sentence/text/tts_text が空です")
            sentence = str(rec.get("sentence") or text)
            tts_text = str(rec.get("tts_text") or apply_readings(sentence, readings))
            records.append({**rec, "sentence": sentence, "tts_text": tts_text})
            continue

        cols = line.split("\t")
        if len(cols) > 1:
            if text_column >= len(cols):
                raise SystemExit(f"{path}:{line_no}: text-column={text_column} が列数を超えています")
            text = cols[text_column].strip()
            term = cols[term_column].strip() if 0 <= term_column < len(cols) else ""
        else:
            text = line
            term = ""
        records.append({"term": term, "sentence": text, "tts_text": apply_readings(text, readings)})
    if not records:
        raise SystemExit(f"入力例文が見つかりません: {path}")
    return records


def main() -> None:
    args = parse_args()
    readings = load_readings(args.readings_tsv)
    records = iter_records(args.input, args.text_column, args.term_column, readings)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for idx, rec in enumerate(records, 1):
            rec = dict(rec)
            rec["id"] = str(rec.get("id") or f"{args.id_prefix}_{idx:04d}")
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} records: {args.out}")


if __name__ == "__main__":
    main()
