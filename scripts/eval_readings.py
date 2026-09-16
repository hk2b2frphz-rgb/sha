#!/usr/bin/env python3
"""
読み仮名生成の定量評価スクリプト (2026-06-13)

generate_sentences.py の出力 JSONL を benchmark_terms.tsv の金標準読みと照合し、
以下の 2 指標を計算する:

  term_replaced    : tts_text 中に元の漢字用語が残っていない割合 (高いほど良い)
  reading_correct  : tts_text 中に金標準の読み仮名が含まれる割合 (高いほど良い)

使い方:
  python scripts/eval_readings.py \
      --sentences out/sentences.jsonl \
      --terms data/benchmark_terms.tsv \
      --model google/gemma-4-E2B-it \
      --out experiments/2026-06-13_e2b_eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def kata_to_hira(text: str) -> str:
    """カタカナをひらがなに変換（比較用）"""
    return "".join(
        chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="読み仮名生成を定量評価する")
    parser.add_argument("--sentences", type=Path, required=True, help="generate_sentences.py の出力 JSONL")
    parser.add_argument("--terms", type=Path, required=True, help="金標準読み TSV (term\\treading)")
    parser.add_argument("--model", required=True, help="使用したモデル名（記録用）")
    parser.add_argument("--out", type=Path, required=True, help="評価結果 JSON の出力先")
    parser.add_argument("--sentences-per-term", type=int, default=None, help="用語あたりの例文数（記録用）")
    return parser.parse_args()


def load_gold(path: Path) -> dict[str, str]:
    """TSV から {term: reading} を読み込む"""
    gold: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("term\t"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            gold[parts[0]] = kata_to_hira(parts[1])
    return gold


def load_sentences(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def evaluate_record(rec: dict, gold: dict[str, str]) -> dict:
    term = rec.get("term", "")
    sentence = rec.get("sentence", "")
    tts_text = rec.get("tts_text", "")
    gold_reading = gold.get(term, "")

    tts_norm = kata_to_hira(tts_text)

    term_replaced = (term not in tts_text) if term else None
    reading_correct = (gold_reading in tts_norm) if gold_reading else None
    tts_changed = tts_text != sentence

    return {
        "id": rec.get("id"),
        "term": term,
        "sentence": sentence,
        "tts_text": tts_text,
        "gold_reading": gold_reading,
        "term_replaced": term_replaced,
        "reading_correct": reading_correct,
        "tts_changed": tts_changed,
    }


def main() -> None:
    args = parse_args()
    gold = load_gold(args.terms)
    records = load_sentences(args.sentences)

    if not records:
        print(f"ERROR: 例文が見つかりません: {args.sentences}", file=sys.stderr)
        sys.exit(1)

    results = [evaluate_record(r, gold) for r in records]

    # 指標計算（gold がある用語のみ）
    with_gold = [r for r in results if r["gold_reading"]]
    n = len(with_gold)
    n_replaced = sum(1 for r in with_gold if r["term_replaced"])
    n_correct = sum(1 for r in with_gold if r["reading_correct"])
    n_changed = sum(1 for r in results if r["tts_changed"])

    acc_replaced = n_replaced / n if n else 0.0
    acc_correct = n_correct / n if n else 0.0

    # 用語別集計
    per_term: dict[str, dict] = {}
    for r in with_gold:
        t = r["term"]
        if t not in per_term:
            per_term[t] = {"term": t, "gold_reading": r["gold_reading"],
                           "n": 0, "n_replaced": 0, "n_correct": 0, "examples": []}
        per_term[t]["n"] += 1
        if r["term_replaced"]:
            per_term[t]["n_replaced"] += 1
        if r["reading_correct"]:
            per_term[t]["n_correct"] += 1
        if len(per_term[t]["examples"]) < 2:
            per_term[t]["examples"].append({"sentence": r["sentence"], "tts_text": r["tts_text"]})

    for v in per_term.values():
        v["acc_replaced"] = v["n_replaced"] / v["n"] if v["n"] else 0.0
        v["acc_correct"] = v["n_correct"] / v["n"] if v["n"] else 0.0

    report = {
        "model": args.model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sentences_file": str(args.sentences),
        "n_total": len(results),
        "n_with_gold": n,
        "n_tts_changed": n_changed,
        "n_term_replaced": n_replaced,
        "n_reading_correct": n_correct,
        "acc_term_replaced": round(acc_replaced, 4),
        "acc_reading_correct": round(acc_correct, 4),
        "per_term": list(per_term.values()),
        "all_results": results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # サマリ表示
    print(f"\n{'='*60}")
    print(f"モデル: {args.model}")
    print(f"{'='*60}")
    print(f"総例文数          : {len(results)}")
    print(f"金標準あり        : {n}")
    print(f"tts_text 変化あり : {n_changed} ({n_changed/len(results)*100:.1f}%)")
    print(f"--- 指標 (金標準ありのみ) ---")
    print(f"term_replaced     : {n_replaced}/{n}  ({acc_replaced*100:.1f}%)")
    print(f"reading_correct   : {n_correct}/{n}  ({acc_correct*100:.1f}%)")
    print(f"{'='*60}")
    print(f"\n用語別:")
    for v in sorted(per_term.values(), key=lambda x: x["acc_correct"]):
        mark = "✓" if v["acc_correct"] >= 1.0 else ("△" if v["acc_correct"] > 0 else "✗")
        print(f"  {mark} {v['term']:20s}  replaced={v['acc_replaced']:.0%}  correct={v['acc_correct']:.0%}  (金標準: {v['gold_reading']})")
    print(f"\n結果を保存: {args.out}")


if __name__ == "__main__":
    main()
