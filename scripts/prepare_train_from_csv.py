#!/usr/bin/env python3
"""generated_sentences.csv -> TTS入力JSONL。

CSV列: Number, Word, Reading, Sentance, AI_Reading, Confirmed_Reading

各行について、Sentance 内の Word を Reading に置換した文を tts_text とし
(専門用語だけをひらがな読みにした発話)、学習ターゲットには元の Sentance
(漢字のまま) を残す。後段の build_whisper_manifest.py が id で join する。

Word が Sentance 内に見つからず置換できない行では、Confirmed_Reading
(文全体をひらがな化した読み) があればそれを tts_text に使う。学習ターゲット
(sentence) は常に元の漢字文のまま。Confirmed_Reading も空の行だけ未解決として
term_mismatch.tsv に記録する。

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
import unicodedata
from pathlib import Path

# CSVヘッダの表記ゆれを吸収する (Sentance は原typo、Sentence も許容)。
COLUMN_ALIASES = {
    "number": ("Number", "number", "No", "no", "id", "ID"),
    "word": ("Word", "word", "term", "Term"),
    "reading": ("Reading", "reading", "yomi", "Yomi"),
    "sentence": ("Sentance", "Sentence", "sentence", "sentance"),
    # 文全体をひらがな化した読み。用語がSentance内に見つからない行のTTSに使う。
    "confirmed_reading": (
        "Confirmed_Reading",
        "confirmed_reading",
        "ConfirmedReading",
        "Confirmed",
        "Conferemed_Reading",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="generated_sentences.csv を TTS入力JSONL へ変換")
    parser.add_argument("--input", type=Path, required=True, help="generated_sentences.csv")
    parser.add_argument("--out", type=Path, required=True, help="出力JSONL")
    parser.add_argument("--id-prefix", default="gs")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="用語不一致の一覧TSV出力先 (既定: <out>と同ディレクトリの term_mismatch.tsv)",
    )
    return parser.parse_args()


def resolve_column(fieldnames: list[str], key: str) -> str:
    for candidate in COLUMN_ALIASES[key]:
        if candidate in fieldnames:
            return candidate
    raise SystemExit(
        f"CSVに必要な列 ({key}) が見つかりません。"
        f" 期待する候補: {COLUMN_ALIASES[key]} / 実際: {fieldnames}"
    )


def resolve_optional_column(fieldnames: list[str], key: str) -> str | None:
    for candidate in COLUMN_ALIASES[key]:
        if candidate in fieldnames:
            return candidate
    return None


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
        col_confirmed = resolve_optional_column(fields, "confirmed_reading")
        rows = list(reader)
    if col_confirmed is None:
        print(
            "[INFO] Confirmed_Reading 列が無いため、用語不一致時のフォールバックは無効です",
            file=sys.stderr,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    missing_term = 0
    used_confirmed = 0
    mismatches: list[tuple[str, str, str, str]] = []  # (id, term, reading, sentence)
    with args.out.open("w", encoding="utf-8") as out_fh:
        for i, row in enumerate(rows, 1):
            sentence = (row.get(col_sentence) or "").strip()
            term = (row.get(col_word) or "").strip()
            reading = (row.get(col_reading) or "").strip()
            confirmed = (row.get(col_confirmed) or "").strip() if col_confirmed else ""
            if not sentence:
                skipped += 1
                continue
            raw_num = (row.get(col_num) or "").strip()
            num = raw_num if raw_num.isdigit() else str(i)
            rec_id = f"{args.id_prefix}_{int(num):04d}"

            if term and reading:
                if term in sentence:
                    tts_text = sentence.replace(term, reading)
                else:
                    # 全角/半角や合成文字(NFC/NFD)の差で in が外れることがある。
                    # NFKC正規化して再照合し、一致すればそちらで置換する。
                    n_term = unicodedata.normalize("NFKC", term)
                    n_sentence = unicodedata.normalize("NFKC", sentence)
                    if n_term in n_sentence:
                        tts_text = n_sentence.replace(n_term, reading)
                        print(
                            f"[INFO] {rec_id}: Word を NFKC正規化後に一致・置換 "
                            f"(元データに全角/半角等の表記ゆれ)",
                            file=sys.stderr,
                        )
                    elif confirmed:
                        # 用語の表層置換が無理でも、文全体のひらがな読み
                        # (Confirmed_Reading) を使えば正しい読みでTTSできる。
                        used_confirmed += 1
                        tts_text = confirmed
                        print(
                            f"[INFO] {rec_id}: Word が発話文に無いため "
                            f"Confirmed_Reading(文全体のひらがな)をTTSに使用",
                            file=sys.stderr,
                        )
                    else:
                        # Confirmed_Reading も無い = 手当てできない。実タイポ判別のため文も出す。
                        missing_term += 1
                        mismatches.append((rec_id, term, reading, sentence))
                        snippet = sentence if len(sentence) <= 50 else sentence[:50] + "..."
                        print(
                            f"[WARN] {rec_id}: Word '{term}' が発話文に見つからず、"
                            f'Confirmed_Reading も空です | 発話文="{snippet}"',
                            file=sys.stderr,
                        )
                        tts_text = sentence
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

    report_path = args.report or (args.out.parent / "term_mismatch.tsv")
    # 不一致の一覧を必ず1ファイルに書き出す (HPCで cat 1発で確認できるように)。
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as rep:
        rep.write("id\tterm\treading\tsentence\n")
        for rec_id, term, reading, sentence in mismatches:
            rep.write(f"{rec_id}\t{term}\t{reading}\t{sentence}\n")

    print(
        f"wrote {written} records -> {args.out} "
        f"(skipped_empty={skipped}, used_confirmed_reading={used_confirmed}, "
        f"unresolved={missing_term})"
    )
    # 末尾サマリ: ログにも一覧を残す。cat する時のパスも明示。
    print("===== TERM MISMATCH SUMMARY =====")
    print(f"mismatches: {len(mismatches)}  report: {report_path}")
    if mismatches:
        for rec_id, term, _reading, sentence in mismatches:
            print(f'  {rec_id}\tWord="{term}"\t発話文="{sentence}"')
        print(f"確認: cat {report_path}")
    else:
        print("  (用語不一致なし)")


if __name__ == "__main__":
    main()
