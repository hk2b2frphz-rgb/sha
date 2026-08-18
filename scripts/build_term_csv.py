#!/usr/bin/env python3
"""用語TSV群 -> generated_sentences.csv (prepare_train_from_csv.py の入力)。

複数ソースの用語TSVを結合し、重複を除き、学習データとして使える行だけを残す。
readingが誤っていると「正しい読みの音声 → 誤った漢字」を学習させることになるので、
機械的に検査できる条件はすべてここで弾く。

入力TSV (タブ区切り, ヘッダ必須): term, reading, sentence, category, source
'＃' または '#' で始まる行はコメント。

Usage:
    python scripts/build_term_csv.py data/terms_*.tsv --out data/generated_sentences.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path

# ひらがな + 長音符・繰り返し記号。読みはこの範囲に収まっていなければならない。
KANA_RE = re.compile(r"^[ぁ-ゖーゝゞ]+$")
# 漢字を含まない用語 (カタカナ語・英字略語) は表記ゆれ学習の対象にならない。
KANJI_RE = re.compile(r"[一-鿿]")


def load_tsv(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "＃"))
    ]
    if not lines:
        return rows
    reader = csv.DictReader(lines, delimiter="\t")
    for lineno, row in enumerate(reader, 2):
        row = {k: (v or "").strip() for k, v in row.items() if k}
        row["_src_file"] = path.name
        row["_lineno"] = str(lineno)
        rows.append(row)
    return rows


def check(row: dict[str, str]) -> str | None:
    """行を弾く理由を返す。問題なければ None。"""
    term, reading, sentence = row.get("term", ""), row.get("reading", ""), row.get("sentence", "")

    if not term or not reading or not sentence:
        return "empty field"
    if not KANJI_RE.search(term):
        return "no kanji (katakana/latin term teaches no spelling)"
    if not KANA_RE.match(reading):
        return "reading is not plain hiragana"
    if term not in sentence:
        # prepare_train_from_csv.py は Sentance 内の Word を Reading に置換する。
        # 一致しなければ置換できず、その行は学習に使えない。
        return "term does not appear verbatim in sentence"
    if reading == term:
        return "reading equals term"
    if len(sentence) < 8:
        return "sentence too short"
    if len(reading) < 2:
        return "reading too short"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, default=None, help="弾いた行の一覧")
    args = ap.parse_args()

    rows: list[dict[str, str]] = []
    for path in args.inputs:
        if not path.exists():
            print(f"[skip] {path} がありません", file=sys.stderr)
            continue
        found = load_tsv(path)
        print(f"[read] {path.name}: {len(found)} 行", file=sys.stderr)
        rows.extend(found)

    kept: list[dict[str, str]] = []
    rejected: list[tuple[dict[str, str], str]] = []
    seen: dict[str, dict[str, str]] = {}
    conflicts: list[tuple[str, str, str]] = []

    for row in rows:
        term = unicodedata.normalize("NFKC", row.get("term", ""))
        row["term"] = term
        reason = check(row)
        if reason:
            rejected.append((row, reason))
            continue
        prev = seen.get(term)
        if prev is not None:
            # 同じ用語に別の読みが付いていたら、どちらかが誤り。両方落として人が見る。
            if prev["reading"] != row["reading"]:
                conflicts.append((term, prev["reading"], row["reading"]))
            continue
        seen[term] = row
        kept.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Number", "Word", "Reading", "Sentance", "AI_Reading", "Confirmed_Reading"])
        for i, row in enumerate(kept, 1):
            writer.writerow([i, row["term"], row["reading"], row["sentence"], "", ""])

    print(f"\n採用 {len(kept)} 件 -> {args.out}", file=sys.stderr)
    print(f"除外 {len(rejected)} 件", file=sys.stderr)
    if conflicts:
        print(f"読みが衝突 {len(conflicts)} 件 (要確認):", file=sys.stderr)
        for term, a, b in conflicts[:10]:
            print(f"  {term}: {a} / {b}", file=sys.stderr)

    by_reason: dict[str, int] = {}
    for _, reason in rejected:
        by_reason[reason] = by_reason.get(reason, 0) + 1
    for reason, n in sorted(by_reason.items(), key=lambda kv: -kv[1]):
        print(f"  {n:5d}  {reason}", file=sys.stderr)

    if args.report:
        with args.report.open("w", encoding="utf-8") as fh:
            for row, reason in rejected:
                fh.write(f"{row['_src_file']}:{row['_lineno']}\t{row.get('term','')}\t{reason}\n")
            for term, a, b in conflicts:
                fh.write(f"CONFLICT\t{term}\t{a} vs {b}\n")
        print(f"詳細 -> {args.report}", file=sys.stderr)


if __name__ == "__main__":
    main()
