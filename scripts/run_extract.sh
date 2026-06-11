#!/usr/bin/env bash
# Run documents (PDF/PPTX/DOCX/images) -> OCR -> technical term list in one go.
#
# Usage:
#   bash scripts/run_extract.sh <docs-dir> [out-dir]
#
# Examples:
#   bash scripts/run_extract.sh docs/            # -> outputs under out/
#   bash scripts/run_extract.sh docs/ out_run2/  # custom output dir
#
# Environment variables:
#   SKIP_OCR=1     skip OCR for text-based PDFs (pass them directly to extract_terms)
#   MIN_COUNT=2    keep only terms appearing in 2+ chunks (noise reduction)
#   MAX_TERMS=100  cap the list at the top 100 most frequent terms
set -euo pipefail

DOCS_DIR="${1:?usage: bash scripts/run_extract.sh <docs-dir> [out-dir]}"
OUT_DIR="${2:-out}"
SKIP_OCR="${SKIP_OCR:-0}"
MIN_COUNT="${MIN_COUNT:-1}"
MAX_TERMS="${MAX_TERMS:-0}"

cd "$(dirname "$0")/.."

echo "=== input: $DOCS_DIR / output: $OUT_DIR ==="

if [ "$SKIP_OCR" = "1" ]; then
    echo "=== STEP 1/2: skipping OCR (SKIP_OCR=1) ==="
    EXTRACT_INPUT="$DOCS_DIR"
else
    echo "=== STEP 1/2: OCR (Qwen3-VL) ==="
    uv run --project gemma_runtime python scripts/ocr_documents.py \
        --docs "$DOCS_DIR" \
        --out-dir "$OUT_DIR/text"
    EXTRACT_INPUT="$OUT_DIR/text"
fi

echo "=== STEP 2/2: term extraction (Gemma 4) ==="
uv run --project gemma_runtime python scripts/extract_terms.py \
    --inputs "$EXTRACT_INPUT" \
    --out "$OUT_DIR/terms.txt" \
    --min-count "$MIN_COUNT" \
    --max-terms "$MAX_TERMS"

echo ""
echo "=== done ==="
echo "term list: $OUT_DIR/terms.txt"
echo "first 20 lines:"
head -n 20 "$OUT_DIR/terms.txt"
echo ""
echo "next step (sentence generation):"
echo "  uv run --project gemma_runtime python scripts/generate_sentences.py \\"
echo "      --terms $OUT_DIR/terms.txt --out $OUT_DIR/sentences.jsonl"
