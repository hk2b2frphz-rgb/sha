#!/usr/bin/env python3
"""Prepare original evaluation sentences for TTS without rewriting all text."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="prepare evaluation JSONL for TTS")
    parser.add_argument("--input", type=Path, required=True, help="one sentence per line, TSV, or JSONL")
    parser.add_argument("--out", type=Path, required=True, help="output JSONL")
    parser.add_argument("--id-prefix", default="eval")
    parser.add_argument("--text-column", type=int, default=0, help="sentence column for TSV input")
    parser.add_argument("--term-column", type=int, default=-1, help="optional term column for TSV input")
    parser.add_argument(
        "--readings-tsv",
        type=Path,
        default=None,
        help="term,reading mapping in tab-separated TSV or comma-separated CSV",
    )
    return parser.parse_args()


def load_readings(path: Path | None) -> list[tuple[str, str]]:
    """Load a term-to-reading mapping from TSV or CSV, longest terms first."""
    if path is None:
        return []
    lines = [
        line
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise SystemExit(f"reading mapping is empty: {path}")
    delimiter = "\t" if any("\t" in line for line in lines) else ","
    rows = list(csv.reader(lines, delimiter=delimiter))
    has_header = len(rows[0]) >= 2 and rows[0][0].strip() == "term" and rows[0][1].strip() == "reading"
    readings: list[tuple[str, str]] = []
    for line_no, cols in enumerate(rows[1 if has_header else 0 :], 2 if has_header else 1):
        if len(cols) < 2:
            raise SystemExit(f"{path}:{line_no}: expected term and reading columns")
        term, reading = cols[0].strip(), cols[1].strip()
        if term and reading:
            readings.append((term, reading))
    if not readings:
        raise SystemExit(f"no term/reading rows in: {path}")
    return sorted(readings, key=lambda item: len(item[0]), reverse=True)


def apply_readings(text: str, readings: list[tuple[str, str]]) -> str:
    for term, reading in readings:
        text = text.replace(term, reading)
    return text


def iter_records(path: Path, text_column: int, term_column: int, readings: list[tuple[str, str]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            rec = json.loads(line)
            sentence = str(rec.get("sentence") or rec.get("text") or rec.get("tts_text") or "").strip()
            if not sentence:
                raise SystemExit(f"{path}:{line_no}: sentence/text/tts_text is empty")
            # Use the original sentence and only apply the requested term map.
            records.append({**rec, "sentence": sentence, "tts_text": apply_readings(sentence, readings)})
            continue

        columns = line.split("\t")
        if len(columns) > 1:
            if text_column >= len(columns):
                raise SystemExit(f"{path}:{line_no}: text-column={text_column} is out of range")
            sentence = columns[text_column].strip()
            term = columns[term_column].strip() if 0 <= term_column < len(columns) else ""
        else:
            sentence, term = line, ""
        records.append({"term": term, "sentence": sentence, "tts_text": apply_readings(sentence, readings)})
    if not records:
        raise SystemExit(f"no input sentences found: {path}")
    return records


def main() -> None:
    args = parse_args()
    readings = load_readings(args.readings_tsv)
    records = iter_records(args.input, args.text_column, args.term_column, readings)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for index, rec in enumerate(records, 1):
            row = dict(rec)
            row["id"] = str(row.get("id") or f"{args.id_prefix}_{index:04d}")
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} records: {args.out}")


if __name__ == "__main__":
    main()
