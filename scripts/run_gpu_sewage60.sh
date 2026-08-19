#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CASES="data/benchmark_terms_sewage_60_cases.tsv"
TODAY="$(date +%F)"
E4B_OUT="experiments/${TODAY}_sewage60_e4b_neweval.json"
E2B_OUT="experiments/${TODAY}_sewage60_e2b_neweval.json"
COMPARISON_JSON="experiments/comparison_results.json"
COMPARISON_MD="experiments/comparison_report.md"

mkdir -p experiments

"$PYTHON_BIN" -m pip install -U pip
"$PYTHON_BIN" -m pip install pyopenjtalk

uv run --project gemma_runtime python scripts/run_eval.py \
  --cases "$CASES" \
  --provider gemma \
  --model google/gemma-4-E4B-it \
  --out "$E4B_OUT"

uv run --project gemma_runtime python scripts/run_eval.py \
  --cases "$CASES" \
  --provider gemma \
  --model google/gemma-4-E2B-it \
  --out "$E2B_OUT"

"$PYTHON_BIN" experiments/run_method_comparison.py \
  --out-json "$COMPARISON_JSON" \
  --out-md "$COMPARISON_MD"

echo "Wrote $E4B_OUT"
echo "Wrote $E2B_OUT"
echo "Wrote $COMPARISON_JSON"
echo "Wrote $COMPARISON_MD"
