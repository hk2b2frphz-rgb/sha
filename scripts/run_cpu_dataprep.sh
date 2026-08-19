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
#   KOKORO_VOICES=...       カンマ区切りのボイス。既定は日本語プリセット5声を巡回する。
#                           単一話者だと decoder がその声の音響に張り付いて実録音に移らない。
#   KOKORO_SPEEDS=...       カンマ区切りの話速。ボイス数と互いに素にして周期を伸ばす。
#   KOKORO_VOICE=jf_alpha   単一ボイス指定 (KOKORO_VOICES 未指定時のみ有効)
#   DEV_RATIO=0.10          dev に回す割合
#   SEED=42                 分割のシード
#   LIMIT=0                 先頭 N 件だけ処理する (0=全件、疎通確認用)
set -euo pipefail

CSV="${1:?usage: bash scripts/run_cpu_dataprep.sh <generated_sentences.csv> [out-dir]}"
OUT_DIR="${2:-out/whisper_turbo}"

KOKORO_VOICE="${KOKORO_VOICE:-jf_alpha}"
KOKORO_VOICES="${KOKORO_VOICES:-jf_alpha,jf_gongitsune,jf_nezumi,jf_tebukuro,jm_kumo}"
KOKORO_SPEEDS="${KOKORO_SPEEDS:-0.92,1.0,1.08}"
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
    --voices "$KOKORO_VOICES" \
    --speeds "$KOKORO_SPEEDS" \
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
echo "=== 生成完了 (GPU 未使用) ==="
echo "train_manifest: $TRAIN_MANIFEST  ($(grep -c . "$TRAIN_MANIFEST") 行)"
echo "wav:            $TTS_OUT/wav"

# ---- 退避: このコンテナは使い捨てなので、置いたままだとセッション終了で消える ----
if [[ -n "${DATA_REPO:-}" ]]; then
    echo ""
    echo "=== HF へ退避: $DATA_REPO ==="
    [[ -n "${HF_TOKEN:-}" ]] || { echo "ERROR: DATA_REPO 指定時は HF_TOKEN が必要です" >&2; exit 1; }
    # --private 既定: 学習用語は社内用語や未公開の固有名詞であることが多く、
    # 公開 repo に上げると用語リストごと外部から読める。公開したい場合は
    # PRIVATE=0 を明示する。
    PRIVATE_FLAG=()
    [[ "${PRIVATE:-1}" == "1" ]] && PRIVATE_FLAG=(--private)
    # 音声は1つの tar にまとめて上げる。個別ファイルのまま上げると、借りた GPU 側で
    # 3800ファイルを1本ずつ取得することになり、実測で数十分かかるうえ、1本でも
    # 取り損ねると学習全体が落ちる (xet 経由の ConnectionError で実際に2回落ちた)。
    # tar なら1ファイルなので取得は数分で済み、失敗の機会も1回に減る。
    echo "音声を tar にまとめています..."
    tar -C "$OUT_DIR" -cf "$OUT_DIR/tts_data.tar" tts_data
    echo "  tts_data.tar: $(du -h "$OUT_DIR/tts_data.tar" | cut -f1)"
    hf upload "$DATA_REPO" "$OUT_DIR" --repo-type dataset "${PRIVATE_FLAG[@]}" \
        --include "train_manifest.jsonl" \
        --include "train.jsonl" \
        --include "dev.jsonl" \
        --include "tts_data.tar"
    echo "退避完了: https://huggingface.co/datasets/$DATA_REPO"
    echo ""
    echo "次: 学習だけ GPU を借りる"
    echo "  HF_TOKEN=\$HF_TOKEN bash scripts/run_whisper_train_vast.sh $DATA_REPO <out-repo>"
else
    echo ""
    echo "!! 警告: このコンテナは使い捨てで、セッションが切れると $OUT_DIR は消えます。"
    echo "   合成し直すと同じ時間がかかるので、必ず HF へ退避してください:"
    echo "     DATA_REPO=<data-repo> HF_TOKEN=hf_xxx bash scripts/run_cpu_dataprep.sh $CSV"
    echo "   すでに生成済みなら手動で:"
    echo "     hf upload <data-repo> $OUT_DIR --repo-type dataset \\"
    echo "         --include 'train_manifest.jsonl' --include 'train.jsonl' \\"
    echo "         --include 'dev.jsonl' --include 'tts_data/**'"
fi
