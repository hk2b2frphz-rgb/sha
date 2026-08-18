#!/usr/bin/env bash
# run_whisper_train.pbs を Vast.ai の時間課金 GPU で回す薄いラッパ。
#
# 学習の中身は複製せず、借りたマシンの上で run_whisper_train.pbs をそのまま
# 実行する (PBS ディレクティブはただのコメントで、PBS_O_WORKDIR / PBS_JOBID は
# どちらもフォールバック付きなので素の bash で動く)。したがって FT_MODE /
# FREEZE_ENCODER / EPOCHS / LR / TTS_BACKEND などの環境変数は PBS 版と同じ意味。
#
# PBS との違い:
#   - GPU を bf16 対応カード (RTX 3090/4090) に限定し MIXED_PRECISION=bf16 を渡す。
#     既定ノードの V100 は bf16 非対応なので PBS 側は fp16 のままにしてある。
#   - NUM_SHARDS 既定は 1。image pull・uv sync・モデル DL の固定時間は GPU 枚数に
#     比例して課金されるため、枚数を増やすとその分だけ割高になる。
#   - 合成済み TTS データを DATA_REPO (HF dataset) に置いて再利用する。TTS は
#     自己回帰生成で最も重い工程なので、2 回目以降のハイパラ変更はここを飛ばせる。
#   - どの終了経路でも必ず instance を destroy する。放置された GPU が一番高い。
#
# Usage:
#   HF_TOKEN=hf_xxx bash scripts/run_whisper_train_vast.sh <hf-data-repo> <hf-out-repo>
#
# Examples:
#   DRY_RUN=1 bash scripts/run_whisper_train_vast.sh me/turbo-tts me/whisper-turbo-ft
#   HF_TOKEN=hf_xxx bash scripts/run_whisper_train_vast.sh me/turbo-tts me/whisper-turbo-ft
#   HF_TOKEN=hf_xxx FT_MODE=lora MAX_HOURS=2 bash scripts/run_whisper_train_vast.sh me/d me/m
#
# DATA_REPO (第1引数) の中身:
#   generated_sentences.csv          学習文の入力 (初回に必要)
#   train_manifest.jsonl, tts_data/  合成済み音声 (初回実行後に自動アップロード)
set -euo pipefail

DATA_REPO="${1:?usage: bash scripts/run_whisper_train_vast.sh <hf-data-repo> <hf-out-repo>}"
OUT_REPO="${2:?usage: bash scripts/run_whisper_train_vast.sh <hf-data-repo> <hf-out-repo>}"

# --- 借りる側の制御 ---
MAX_HOURS="${MAX_HOURS:-4}"
MAX_DPH="${MAX_DPH:-0.25}"
GPU="${GPU:-RTX_3090,RTX_4090}"
DISK="${DISK:-60}"
SPOT="${SPOT:-0}"
DRY_RUN="${DRY_RUN:-0}"
IMAGE="${IMAGE:-pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel}"

# --- run_whisper_train.pbs にそのまま渡す変数 ---
NUM_SHARDS="${NUM_SHARDS:-1}"
BASE_MODEL="${BASE_MODEL:-openai/whisper-large-v3-turbo}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
PASSTHROUGH=(FT_MODE FREEZE_ENCODER TTS_BACKEND KOKORO_VOICE KOKORO_MODEL
             TTS_SPEAKER TTS_MODEL TTS_DTYPE EPOCHS LR BATCH_SIZE GRAD_ACCUM
             LORA_R LORA_ALPHA LANGUAGE CT2_QUANT FORCE_REBUILD_DATA PROGRESS_EVERY)

cd "$(dirname "$0")/.."
BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
REPO_URL="${REPO_URL:-$(git remote get-url origin)}"

if [[ "$DRY_RUN" != "1" && -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN が必要です ($DATA_REPO の取得と $OUT_REPO への書き戻しに使う)" >&2
    exit 1
fi
command -v vastai >/dev/null || { echo "vastai CLI がありません: pip install --user vastai" >&2; exit 1; }

# --- 最安の提示を選ぶ -------------------------------------------------------
# inet_down を高めに要求する: turbo の重みと Kokoro/unidic の DL 時間も課金対象。
QUERY="gpu_name in [${GPU}] num_gpus>=${NUM_SHARDS} disk_space>=${DISK} inet_down>=300 reliability>0.98 rentable=true"

echo "=== 提示を検索: $QUERY ==="
PICKED="$(vastai search offers "$QUERY" -o 'dph+' --raw | MAX_DPH="$MAX_DPH" python3 -c '
import json, os, sys
cap = float(os.environ["MAX_DPH"])
for o in json.load(sys.stdin):
    if o["dph_total"] <= cap:
        print(o["id"], "%.4f" % o["dph_total"], o["gpu_name"].replace(" ", "_"))
        break
else:
    sys.exit("$%.3f/hr 以下の提示なし -- MAX_DPH を上げてください" % cap)
')"
read -r OFFER_ID OFFER_DPH OFFER_GPU <<<"$PICKED"

echo "offer $OFFER_ID: $OFFER_GPU  \$$OFFER_DPH/hr"
echo "MAX_HOURS=$MAX_HOURS 時点の最悪支出: \$$(python3 -c "print(f'{$OFFER_DPH * $MAX_HOURS:.2f}')")"

# PBS 版と同じ変数を、呼び出し側で設定されているものだけ引き継ぐ
ENV_LINES="export BASE_MODEL='$BASE_MODEL' MIXED_PRECISION='$MIXED_PRECISION' NUM_SHARDS='$NUM_SHARDS'"
for v in "${PASSTHROUGH[@]}"; do
    [[ -n "${!v:-}" ]] && ENV_LINES="$ENV_LINES $v='${!v}'"
done
echo "引き継ぐ設定: $ENV_LINES"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN=1 のため借りずに終了。"
    exit 0
fi

# --- 借りたマシンで動かす中身 -----------------------------------------------
read -r -d '' ONSTART <<ONSTART_EOF || true
set -eux
export HF_HUB_ENABLE_HF_TRANSFER=1 DEBIAN_FRONTEND=noninteractive
pip install -q uv "huggingface_hub[cli,hf_transfer]"
git clone --depth 1 --branch "$BRANCH" "$REPO_URL" /workspace/repo
cd /workspace/repo

WORK_ROOT=out/whisper_turbo
mkdir -p "\$WORK_ROOT"

# 合成済みデータを取ってくる。TTS は CPU コンテナ側で済ませておく前提なので、
# ここに無ければ即座に失敗させる (借りた GPU で TTS を回すのが一番もったいない)。
hf download "$DATA_REPO" --repo-type dataset --local-dir "\$WORK_ROOT"
test -s "\$WORK_ROOT/train_manifest.jsonl" || {
    echo "ERROR: train_manifest.jsonl が $DATA_REPO にありません。" >&2
    echo "       先に CPU 側で: bash scripts/run_cpu_dataprep.sh <csv>" >&2
    exit 1
}

# manifest の audio は生成元マシンの絶対パス。このマシンの実体に貼り替える。
python3 scripts/rebase_manifest_paths.py \
    --manifest "\$WORK_ROOT/train_manifest.jsonl" \
    --root "\$WORK_ROOT" --in-place

$ENV_LINES
export WORK_ROOT
# train_manifest.jsonl があるので run_whisper_train.pbs は step 1 の TTS を飛ばす。
bash scripts/run_whisper_train.pbs

hf upload "$OUT_REPO" "\$WORK_ROOT/ct2" --repo-type model
echo "TRAINING_COMPLETE"
ONSTART_EOF

# --- 借りたら必ず返す -------------------------------------------------------
CREATE_ARGS=(create instance "$OFFER_ID"
    --image "$IMAGE"
    --disk "$DISK"
    --env "-e HF_TOKEN=$HF_TOKEN -e HF_HUB_ENABLE_HF_TRANSFER=1"
    --onstart-cmd "$ONSTART"
    --raw)
[[ "$SPOT" == "1" ]] && CREATE_ARGS+=(--bid "$OFFER_DPH")

INSTANCE_ID="$(vastai "${CREATE_ARGS[@]}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["new_contract"])')"
echo "=== instance $INSTANCE_ID を借りた ==="

destroy() {
    echo ""
    echo "=== instance $INSTANCE_ID を破棄 ==="
    for _ in 1 2 3 4 5; do
        vastai destroy instance "$INSTANCE_ID" && return 0
        sleep 5
    done
    echo "!! $INSTANCE_ID を破棄できませんでした。今すぐ手で消してください:" >&2
    echo "   vastai destroy instance $INSTANCE_ID" >&2
    return 1
}
trap destroy EXIT INT TERM

# --- 完了・エラー・予算切れのいずれかまで見張る -----------------------------
# ローカル側の締切は MAX_HOURS + 固定作業分 (image pull / uv sync / モデル DL /
# 結果アップロード)。TTS 初回はここが伸びるので余裕を多めに取る。
DEADLINE=$(python3 -c "import time; print(int(time.time() + $MAX_HOURS * 3600 + 3600))")
echo "=== 監視中 (Ctrl-C でも破棄されます) ==="

while [[ "$(date +%s)" -lt "$DEADLINE" ]]; do
    sleep 60
    LOG="$(vastai logs "$INSTANCE_ID" --tail 60 2>/dev/null || true)"
    printf '%s' "$LOG" | grep -E "tts-progress|step [0-9]/4|'loss'|\[data\]" | tail -2 || true

    if printf '%s' "$LOG" | grep -q TRAINING_COMPLETE; then
        echo ""
        echo "=== 完了 -> https://huggingface.co/$OUT_REPO ==="
        exit 0
    fi
    if printf '%s' "$LOG" | grep -qE "Traceback|CUDA out of memory|^ERROR:"; then
        echo ""
        echo "=== インスタンス側でエラー。破棄して終了します ===" >&2
        vastai logs "$INSTANCE_ID" --tail 120 >&2 || true
        exit 1
    fi
done

echo "=== ローカル締切に到達。破棄します ===" >&2
exit 1
