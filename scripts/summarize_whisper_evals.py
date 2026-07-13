#!/usr/bin/env python3
"""Collect per-model Whisper evaluation summaries into a comparable ranking."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="summarize batch Whisper evaluations")
    parser.add_argument("--models-file", type=Path, required=True, help="TSV: label<TAB>CT2 model directory")
    parser.add_argument("--results-dir", type=Path, required=True, help="directory containing <label>/summary.json")
    return parser.parse_args()


def read_models(path: Path) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        columns = raw.split("\t")
        if len(columns) != 2 or not columns[0].strip() or not columns[1].strip():
            raise SystemExit(f"{path}:{line_no}: expected 'label<TAB>model_dir'")
        label = columns[0].strip()
        if label in seen:
            raise SystemExit(f"{path}:{line_no}: duplicate label: {label}")
        seen.add(label)
        labels.append(label)
    if not labels:
        raise SystemExit(f"no models found in: {path}")
    return labels


def main() -> None:
    args = parse_args()
    labels = read_models(args.models_file)
    rows: list[dict[str, object]] = []
    for label in labels:
        summary_path = args.results_dir / label / "summary.json"
        if not summary_path.is_file():
            raise SystemExit(f"summary not found (evaluation failed or was skipped): {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model": label,
                "cer": float(summary["cer"]),
                "wer": float(summary["wer"]),
                "speed_x": float(summary["speed_x"]),
                "rtf": float(summary["rtf"]),
                "n": int(summary["n"]),
                "model_dir": str(summary["model_dir"]),
            }
        )
    rows.sort(key=lambda row: (float(row["cer"]), float(row["wer"]), -float(row["speed_x"])))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank

    args.results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.results_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["rank", "model", "cer", "wer", "speed_x", "rtf", "n", "model_dir"])
        writer.writeheader()
        writer.writerows(rows)

    report_path = args.results_dir / "summary.md"
    report_lines = [
        "# Whisper model comparison",
        "",
        "Lower CER/WER is better; higher speed is better. Rankings prioritize CER, then WER, then speed.",
        "",
        "| Rank | Model | CER | WER | Speed | RTF | Samples |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        report_lines.append(
            f"| {row['rank']} | {row['model']} | {row['cer']:.4f} | {row['wer']:.4f} | "
            f"{row['speed_x']:.2f}x | {row['rtf']:.4f} | {row['n']} |"
        )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"comparison: {report_path}")
    print(f"csv:        {csv_path}")


if __name__ == "__main__":
    main()
