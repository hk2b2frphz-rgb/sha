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
# run_whisper_train.pbs と同じ既定。checkpoint の置き場 (ADAPTER_ROOT) が
# WORK_ROOT/$FT_MODE なので、こちら側でも確定させておく必要がある。
FT_MODE="${FT_MODE:-full}"
BASE_MODEL="${BASE_MODEL:-openai/whisper-large-v3-turbo}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
# 借りたマシンは終了時に破棄されるので、checkpoint は毎回 HF へ退避し、
# 起動時に取り戻して再開する。既定は <out-repo>-ckpt。
CHECKPOINT_REPO="${CHECKPOINT_REPO:-${OUT_REPO}-ckpt}"
RESUME="${RESUME:-1}"
# 学習済みモデルは学習に使った用語を復元できる。既定は非公開。
PRIVATE="${PRIVATE:-1}"
PRIVATE_FLAG=""
[[ "$PRIVATE" == "1" ]] && PRIVATE_FLAG="--private"
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

# $1 に除外したい offer id をカンマ区切りで渡す (起動に失敗したホストの再選択を避ける)。
pick_offer() {
    local picked
    picked="$(vastai search offers "$QUERY" -o 'dph+' --raw \
        | MAX_DPH="$MAX_DPH" SKIP_IDS="${1:-}" python3 -c '
import json, os, sys
cap = float(os.environ["MAX_DPH"])
skip = {x for x in os.environ.get("SKIP_IDS", "").split(",") if x}
for o in json.load(sys.stdin):
    if o["dph_total"] <= cap and str(o["id"]) not in skip:
        print(o["id"], "%.4f" % o["dph_total"], o["gpu_name"].replace(" ", "_"))
        break
else:
    sys.exit("$%.3f/hr 以下の未試行の提示なし -- MAX_DPH を上げてください" % cap)
')"
    read -r OFFER_ID OFFER_DPH OFFER_GPU <<<"$picked"
    echo "offer $OFFER_ID: $OFFER_GPU  \$$OFFER_DPH/hr"
    echo "MAX_HOURS=$MAX_HOURS 時点の最悪支出: \$$(python3 -c "print(f'{$OFFER_DPH * $MAX_HOURS:.2f}')")"
}
pick_offer ""

# PBS 版と同じ変数を、呼び出し側で設定されているものだけ引き継ぐ
ENV_LINES="export BASE_MODEL='$BASE_MODEL' MIXED_PRECISION='$MIXED_PRECISION' NUM_SHARDS='$NUM_SHARDS'"
ENV_LINES="$ENV_LINES CHECKPOINT_REPO='$CHECKPOINT_REPO' RESUME='$RESUME'"
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
# 失敗の検知をログ本文のパターン照合に頼らない。set -e で落ちた場合も
# 明示的な exit の場合も、必ず ONSTART_FAILED を出してから終わる。
# (v2 では HF の ConnectionError が "Traceback|ERROR:" のどれにも当たらず、
#  監視側が失敗に気付けないまま 2.5 時間分課金された)
_ONSTART_DONE=0
trap '[ "\$_ONSTART_DONE" = 1 ] || echo "ONSTART_FAILED rc=\$?"' EXIT
export DEBIAN_FRONTEND=noninteractive
# hf_transfer を「有効化だけして未インストール」にすると、huggingface_hub は
# ダウンロード時に ValueError を投げて学習が落ちる (実際に落ちた)。
# extra 指定 [hf_transfer] は現行版で廃止され警告のみで無視されるので、
# パッケージを直接入れ、入った場合にだけ有効化する。
pip install -q uv huggingface_hub
python -c "import hf_transfer" 2>/dev/null && export HF_HUB_ENABLE_HF_TRANSFER=1 || true
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

# 前回が中断されていれば checkpoint を取り戻す (無ければ何も起きない)。
# 置き場は ADAPTER_ROOT=\$WORK_ROOT/\$FT_MODE で、そこの trainer/ を見て再開する。
hf download "$CHECKPOINT_REPO" --local-dir "\$WORK_ROOT/$FT_MODE" 2>/dev/null \
    && echo "[ckpt] 前回の checkpoint を取得した" \
    || echo "[ckpt] 既存 checkpoint なし (新規学習)"

$ENV_LINES
export WORK_ROOT
# train_manifest.jsonl があるので run_whisper_train.pbs は step 1 の TTS を飛ばす。
bash scripts/run_whisper_train.pbs

hf upload "$OUT_REPO" "\$WORK_ROOT/ct2" --repo-type model $PRIVATE_FLAG
_ONSTART_DONE=1
echo "TRAINING_COMPLETE"
ONSTART_EOF

# --- 借りたら必ず返す -------------------------------------------------------
INSTANCE_ID=""
destroy() {
    [[ -n "$INSTANCE_ID" ]] || return 0
    echo ""
    echo "=== instance $INSTANCE_ID を破棄 ==="
    for _ in 1 2 3 4 5; do
        # -y は必須: これが無いと対話確認 ([y/N]) で止まり、非対話実行では
        # Aborted になって課金が続く。実際にそれでインスタンスが生き残った。
        if vastai destroy instance -y "$INSTANCE_ID"; then
            INSTANCE_ID=""
            return 0
        fi
        sleep 5
    done
    echo "!! $INSTANCE_ID を破棄できませんでした。今すぐ手で消してください:" >&2
    echo "   vastai destroy instance -y $INSTANCE_ID" >&2
    return 1
}
trap destroy EXIT INT TERM

rent_instance() {
    local create_args=(create instance "$OFFER_ID"
        --image "$IMAGE"
        --disk "$DISK"
        --env "-e HF_TOKEN=$HF_TOKEN"
        --onstart-cmd "$ONSTART"
        --raw)
    [[ "$SPOT" == "1" ]] && create_args+=(--bid "$OFFER_DPH")
    INSTANCE_ID="$(vastai "${create_args[@]}" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["new_contract"])')"
    echo "=== instance $INSTANCE_ID を借りた ==="
}

# --- 完了・エラー・予算切れのいずれかまで見張る -----------------------------
# ローカル側の締切は MAX_HOURS + 固定作業分 (image pull / uv sync / モデル DL /
# 結果アップロード)。TTS 初回はここが伸びるので余裕を多めに取る。
# --- 完了・エラー・予算切れのいずれかまで見張る -----------------------------
# ログが1行も動かない状態が続いたら落ちたとみなす。番兵を出さずに死ぬ経路
# (OOM killer / ホスト側の停止) を拾うための保険。
STALL_MINUTES="${STALL_MINUTES:-25}"
# 借りたのにホストがコンテナを起動できないことがある (実際に発生し、
# "No such container" のまま何も起きなかった)。これはホスト固有の障害なので、
# 短く見切って別の提示で借り直す。
START_TIMEOUT_MIN="${START_TIMEOUT_MIN:-8}"
ATTEMPTS="${ATTEMPTS:-3}"
# vastai logs 自体が続けて失敗したら監視できていないので打ち切る (分)。
TRANSIENT_LIMIT="${TRANSIENT_LIMIT:-15}"

# 0=完了 / 1=打ち切り (このホストの問題ではない) / 2=ホストが起動しない
monitor_instance() {
    local deadline started_at now sig log err rc transient=0
    local seen_output=0 last_sig="" last_change errfile
    started_at="$(date +%s)"
    last_change="$started_at"
    errfile="$(mktemp)"
    # ローカル側の締切は MAX_HOURS + 固定作業分 (image pull / uv sync / モデル DL /
    # 結果アップロード)。
    deadline=$(python3 -c "import time; print(int(time.time() + $MAX_HOURS * 3600 + 3600))")

    while [[ "$(date +%s)" -lt "$deadline" ]]; do
        sleep 60
        # stdout と stderr を分けて受ける。インスタンスのログ本文は stdout に、
        # vastai CLI 自身の失敗 (S3 からのログ取得タイムアウト等) は stderr に出る。
        # 以前ここを 2>&1 でまとめてしまい、CLI 自身の Traceback を「インスタンス側の
        # エラー」と誤判定して健全なインスタンスを破棄した。
        rc=0
        log="$(vastai logs "$INSTANCE_ID" --tail 60 2>"$errfile")" || rc=$?
        err="$(cat "$errfile")"

        # ホストがコンテナを起動できていない場合のメッセージだけは stderr も見る。
        if printf '%s\n%s' "$log" "$err" | grep -q "No such container"; then
            now="$(date +%s)"
            if (( now - started_at > START_TIMEOUT_MIN * 60 )); then
                echo "" >&2
                echo "=== ${START_TIMEOUT_MIN}分たってもコンテナが起動しません。別のホストで借り直します ===" >&2
                rm -f "$errfile"
                return 2
            fi
            continue
        fi

        # CLI 側の一時的な失敗。インスタンスの状態は不明なので、成否の判定には
        # 使わずに次の周回へ回す。ただし読めない状態が続くなら監視できていないので、
        # 課金を垂れ流さないよう打ち切る。
        if (( rc != 0 )); then
            transient=$((transient + 1))
            echo "[warn] vastai logs が失敗 (${transient}/${TRANSIENT_LIMIT}): $(printf '%s' "$err" | tail -1)" >&2
            if (( transient >= TRANSIENT_LIMIT )); then
                echo "=== ログを${TRANSIENT_LIMIT}回続けて取得できません。破棄して終了します ===" >&2
                rm -f "$errfile"
                return 1
            fi
            last_change="$(date +%s)"   # 状態不明の間はストール判定を進めない
            continue
        fi
        transient=0

        printf '%s' "$log" | grep -E "tts-progress|step [0-9]/4|'loss'|\[data\]" | tail -2 || true

        if printf '%s' "$log" | grep -q TRAINING_COMPLETE; then
            rm -f "$errfile"
            return 0
        fi
        # ONSTART_FAILED が主。残りは番兵より先に気付けたとき用の早期打ち切り。
        # 判定は必ず stdout (インスタンスのログ本文) だけに対して行う。
        if printf '%s' "$log" | grep -qE "ONSTART_FAILED|Traceback|CUDA out of memory|^ERROR:"; then
            echo "" >&2
            echo "=== インスタンス側でエラー。破棄して終了します ===" >&2
            vastai logs "$INSTANCE_ID" --tail 120 >&2 || true
            rm -f "$errfile"
            return 1
        fi

        now="$(date +%s)"
        [[ -n "$log" ]] && seen_output=1
        if (( seen_output == 0 && now - started_at > START_TIMEOUT_MIN * 60 )); then
            echo "" >&2
            echo "=== ${START_TIMEOUT_MIN}分たってもログが出ません。別のホストで借り直します ===" >&2
            rm -f "$errfile"
            return 2
        fi

        sig="$(printf '%s' "$log" | md5sum | cut -d' ' -f1)"
        if [[ "$sig" != "$last_sig" ]]; then
            last_sig="$sig"
            last_change="$now"
        elif (( now - last_change > STALL_MINUTES * 60 )); then
            echo "" >&2
            echo "=== ${STALL_MINUTES}分ログが動きません。破棄して終了します ===" >&2
            vastai logs "$INSTANCE_ID" --tail 120 >&2 || true
            rm -f "$errfile"
            return 1
        fi
    done
    rm -f "$errfile"
    echo "=== ローカル締切に到達。破棄します ===" >&2
    return 1
}

# 起動に失敗したホストは除外して借り直す。checkpoint は HF にあるので、
# 途中まで進んでいれば RESUME=1 で続きから再開される。
TRIED=""
STATUS=1
for attempt in $(seq 1 "$ATTEMPTS"); do
    if [[ "$attempt" -gt 1 ]]; then
        echo ""
        echo "=== 別のホストで再試行 ($attempt/$ATTEMPTS) ==="
        pick_offer "$TRIED"
    fi
    TRIED="${TRIED:+$TRIED,}$OFFER_ID"
    rent_instance
    echo "=== 監視中 (Ctrl-C でも破棄されます / 起動 ${START_TIMEOUT_MIN}分・無進捗 ${STALL_MINUTES}分で打ち切り) ==="
    if monitor_instance; then STATUS=0; else STATUS=$?; fi
    [[ "$STATUS" == 2 ]] || break
    destroy
done

if [[ "$STATUS" == 0 ]]; then
    echo ""
    echo "=== 完了 -> https://huggingface.co/$OUT_REPO ==="
    exit 0
fi
exit 1
