#!/usr/bin/env bash
# CPU だけで学習データを作る。GPU は使わない。
#
# generated_sentences.csv -> TTS入力 -> Kokoro合成 -> train/dev manifest まで。
# 学習だけを Vast.ai の GPU に投げる構成のうち、CPU 側の全工程にあたる。
#
# Kokoro-82M は小さく、このコンテナ (4 vCPU) で実時間より速く合成できるため
# TTS に GPU を借りる必要がない。借りるのは学習の数時間だけで済む。
#
# Usage:
#   bash scripts/run_cpu_dataprep.sh <generated_sentences.csv> [out-dir]
#
# Environment variables:
#   KOKORO_VOICE=jf_alpha   Kokoro 日本語ボイス (jf_alpha/jf_gongitsune/jf_tebukuro/jm_kumo)
#   DEV_RATIO=0.10          dev に回す割合
#   SEED=42                 分割のシード
#   LIMIT=0                 先頭 N 件だけ処理する (0=全件、疎通確認用)
set -euo pipefail

CSV="${1:?usage: bash scripts/run_cpu_dataprep.sh <generated_sentences.csv> [out-dir]}"
OUT_DIR="${2:-out/whisper_turbo}"

KOKORO_VOICE="${KOKORO_VOICE:-jf_alpha}"
KOKORO_MODEL="${KOKORO_MODEL:-hexgrad/Kokoro-82M}"
DEV_RATIO="${DEV_RATIO:-0.10}"
SEED="${SEED:-42}"
LIMIT="${LIMIT:-0}"

cd "$(dirname "$0")/.."
VENV="${VENV:-.venv-cpu}"
[[ -d "$VENV" ]] || { echo "ERROR: $VENV がありません。bash scripts/setup_cpu_env.sh を先に実行してください。" >&2; exit 1; }
# shellcheck source=/dev/null
source "$VENV/bin/activate"

TTS_INPUT="$OUT_DIR/tts_input.jsonl"
TTS_OUT="$OUT_DIR/tts_data/shard_00"
TRAIN_MANIFEST="$OUT_DIR/train_manifest.jsonl"

mkdir -p "$OUT_DIR"
echo "=== 入力: $CSV / 出力: $OUT_DIR (CPU only) ==="

# ---- step 1/4: CSV -> TTS入力 ---------------------------------------------
echo "=== step 1/4: CSV -> TTS入力 ==="
python scripts/prepare_train_from_csv.py --input "$CSV" --out "$TTS_INPUT"

if [[ "$LIMIT" != "0" ]]; then
    echo "LIMIT=$LIMIT: 先頭 $LIMIT 件に絞る"
    head -n "$LIMIT" "$TTS_INPUT" > "$TTS_INPUT.tmp" && mv "$TTS_INPUT.tmp" "$TTS_INPUT"
fi
echo "対象: $(grep -c . "$TTS_INPUT") 文"

# ---- step 2/4: Kokoro で合成 (CPU) ----------------------------------------
# Kokoro は misaki[ja] が full unidic を要求し、本体の unidic-lite と衝突するため
# 隔離 env で動かす (run_whisper_train.pbs と同じ理由)。torch は CPU ビルドを使う。
echo "=== step 2/4: Kokoro TTS (CPU) ==="
TORCH_VERSION="$(python -c 'import torch; print(torch.__version__.split("+")[0])')"
TORCHAUDIO_VERSION="$(python -c 'import torchaudio; print(torchaudio.__version__.split("+")[0])')"
KOKORO_UV=(
    uv run --isolated --no-project
    --with "kokoro>=0.9.4"
    --with "misaki[ja]"
    --with unidic
    --with pyopenjtalk
    --with soundfile
    --with numpy
    --with "torch==$TORCH_VERSION"
    --with "torchaudio==$TORCHAUDIO_VERSION"
)
"${KOKORO_UV[@]}" python -m unidic download >/dev/null 2>&1 || true
"${KOKORO_UV[@]}" python scripts/synthesize_speech_kokoro.py \
    --sentences "$TTS_INPUT" \
    --out-dir "$TTS_OUT" \
    --model-id "$KOKORO_MODEL" \
    --voice "$KOKORO_VOICE" \
    --device cpu

# ---- step 3/4: 学習 manifest -----------------------------------------------
echo "=== step 3/4: 学習 manifest ==="
python scripts/build_whisper_manifest.py \
    --tts-input "$TTS_INPUT" \
    --synth-dir "$TTS_OUT" \
    --out "$TRAIN_MANIFEST" \
    --summary "$OUT_DIR/manifest_summary.json"

# ---- step 4/4: train / dev 分割 --------------------------------------------
echo "=== step 4/4: train/dev 分割 ==="
python scripts/split_whisper_manifest.py \
    --manifest "$TRAIN_MANIFEST" \
    --train-out "$OUT_DIR/train.jsonl" \
    --dev-out "$OUT_DIR/dev.jsonl" \
    --summary "$OUT_DIR/split_summary.json" \
    --dev-ratio "$DEV_RATIO" \
    --seed "$SEED"

echo ""
echo "=== 完了 (GPU 未使用) ==="
echo "train_manifest: $TRAIN_MANIFEST  ($(grep -c . "$TRAIN_MANIFEST") 行)"
echo "wav:            $TTS_OUT/wav"
echo ""
echo "次: HF にアップロードして学習だけ GPU を借りる"
echo "  hf upload <data-repo> $OUT_DIR --repo-type dataset \\"
echo "      --include 'train_manifest.jsonl' --include 'train.jsonl' --include 'dev.jsonl' --include 'tts_data/**'"
echo "  HF_TOKEN=hf_xxx bash scripts/run_whisper_train_vast.sh <data-repo> <out-repo>"
