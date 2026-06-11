#!/usr/bin/env bash
# 文書 (PDF/PPTX/DOCX/画像) → OCR → 専門用語リストまでを一気に実行する。
#
# 使い方:
#   bash scripts/run_extract.sh <文書ディレクトリ> [出力ディレクトリ]
#
# 例:
#   bash scripts/run_extract.sh docs/            # -> out/ 配下に出力
#   bash scripts/run_extract.sh docs/ out_run2/  # 出力先を変える
#
# 環境変数で調整:
#   SKIP_OCR=1     テキスト PDF のみで OCR が不要な場合 (extract_terms に直接渡す)
#   MIN_COUNT=2    2 チャンク以上に出現した用語のみ採用 (ノイズ削減)
#   MAX_TERMS=100  頻出順に上位 100 語へ絞る
set -euo pipefail

DOCS_DIR="${1:?usage: bash scripts/run_extract.sh <docs-dir> [out-dir]}"
OUT_DIR="${2:-out}"
SKIP_OCR="${SKIP_OCR:-0}"
MIN_COUNT="${MIN_COUNT:-1}"
MAX_TERMS="${MAX_TERMS:-0}"

cd "$(dirname "$0")/.."

echo "=== 入力: $DOCS_DIR / 出力: $OUT_DIR ==="

if [ "$SKIP_OCR" = "1" ]; then
    echo "=== STEP 1/2: OCR をスキップ (SKIP_OCR=1) ==="
    EXTRACT_INPUT="$DOCS_DIR"
else
    echo "=== STEP 1/2: OCR (Qwen3-VL) ==="
    uv run --project gemma_runtime python scripts/ocr_documents.py \
        --docs "$DOCS_DIR" \
        --out-dir "$OUT_DIR/text"
    EXTRACT_INPUT="$OUT_DIR/text"
fi

echo "=== STEP 2/2: 専門用語抽出 (Gemma 4) ==="
uv run --project gemma_runtime python scripts/extract_terms.py \
    --inputs "$EXTRACT_INPUT" \
    --out "$OUT_DIR/terms.txt" \
    --min-count "$MIN_COUNT" \
    --max-terms "$MAX_TERMS"

echo ""
echo "=== 完了 ==="
echo "用語リスト: $OUT_DIR/terms.txt"
echo "先頭 20 行:"
head -n 20 "$OUT_DIR/terms.txt"
echo ""
echo "次のステップ (例文生成):"
echo "  uv run --project gemma_runtime python scripts/generate_sentences.py \\"
echo "      --terms $OUT_DIR/terms.txt --out $OUT_DIR/sentences.jsonl"
