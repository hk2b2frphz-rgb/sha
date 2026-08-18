#!/usr/bin/env bash
# Rent the cheapest suitable GPU on Vast.ai, fine-tune Whisper, destroy the box.
#
# The instance is destroyed on every exit path -- normal finish, training
# failure, Ctrl-C, or the local timeout -- because idle rented GPUs are where
# the money actually goes ($0.11/h left running overnight is ~13% of a $10
# balance).
#
# Usage:
#   bash scripts/vast_train.sh <hf-dataset-repo> <hf-output-repo>
#
# Examples:
#   bash scripts/vast_train.sh myname/term2speech-audio myname/whisper-ja-terms
#   MAX_HOURS=2 GPU='RTX_4090' bash scripts/vast_train.sh me/data me/model
#   DRY_RUN=1 bash scripts/vast_train.sh me/data me/model   # price check only
#
# Environment variables:
#   HF_TOKEN     (required) write token; the box pulls data and pushes results
#   MAX_HOURS=3  hard cap passed to the trainer AND to the local watchdog
#   MAX_DPH=0.20 refuse offers above this $/hour
#   GPU=...      comma-separated allowlist (default: bf16-capable 24 GB cards)
#   DISK=40      GB of instance disk to rent
#   SPOT=1       bid on an interruptible instance (~15% cheaper, preemptible)
#   BRANCH=...   branch of this repo to train from (default: current branch)
#   DRY_RUN=1    show the chosen offer and exit without renting
set -euo pipefail

DATA_REPO="${1:?usage: bash scripts/vast_train.sh <hf-dataset-repo> <hf-output-repo>}"
OUT_REPO="${2:?usage: bash scripts/vast_train.sh <hf-dataset-repo> <hf-output-repo>}"

MAX_HOURS="${MAX_HOURS:-3}"
MAX_DPH="${MAX_DPH:-0.20}"
GPU="${GPU:-RTX_3090,RTX_4090}"
DISK="${DISK:-40}"
SPOT="${SPOT:-0}"
DRY_RUN="${DRY_RUN:-0}"
IMAGE="${IMAGE:-pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel}"

cd "$(dirname "$0")/.."
BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
REPO_URL="${REPO_URL:-$(git remote get-url origin)}"

if [ "$DRY_RUN" != "1" ] && [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN is required: the instance needs it to pull $DATA_REPO and push $OUT_REPO" >&2
    exit 1
fi

command -v vastai >/dev/null || { echo "vastai CLI not found; pip install --user vastai" >&2; exit 1; }

# --- pick the cheapest offer -------------------------------------------------
# Ask for more inet_down than strictly needed: large-v3 is a ~3 GB download and
# you are billed for every second of it.
QUERY="gpu_name in [${GPU}] num_gpus=1 disk_space>=${DISK} inet_down>=200 reliability>0.98 rentable=true"
if [ "$SPOT" = "1" ]; then
    QUERY="$QUERY rented=false"
fi

echo "=== searching offers: $QUERY ==="
OFFER_JSON="$(vastai search offers "$QUERY" -o 'dph+' --raw)"

PICKED="$(printf '%s' "$OFFER_JSON" | MAX_DPH="$MAX_DPH" python3 -c '
import json, os, sys
cap = float(os.environ["MAX_DPH"])
for o in json.load(sys.stdin):
    if o["dph_total"] <= cap:
        print(o["id"], "%.4f" % o["dph_total"], o["gpu_name"].replace(" ", "_"))
        break
else:
    sys.exit("no offer at or under $%.3f/hr -- raise MAX_DPH" % cap)
')"
read -r OFFER_ID OFFER_DPH OFFER_GPU <<<"$PICKED"

EST_COST="$(python3 -c "print(f'{$OFFER_DPH * $MAX_HOURS:.2f}')")"
echo "chosen offer $OFFER_ID: $OFFER_GPU at \$$OFFER_DPH/hr"
echo "worst-case spend at MAX_HOURS=$MAX_HOURS: \$$EST_COST"

if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN=1, not renting."
    exit 0
fi

# --- what the box runs on boot ----------------------------------------------
ONSTART=$(cat <<ONSTART_EOF
set -eux
export HF_HUB_ENABLE_HF_TRANSFER=1
pip install -q --upgrade transformers peft accelerate bitsandbytes datasets \
    soundfile torchaudio ctranslate2 hf_transfer "huggingface_hub[cli]"
git clone --depth 1 --branch "$BRANCH" "$REPO_URL" /workspace/repo
cd /workspace/repo
hf download "$DATA_REPO" --repo-type dataset --local-dir /workspace/data
python scripts/finetune_whisper.py \
    --manifest /workspace/data/manifest.jsonl \
    --out-dir /workspace/ft \
    --max-hours $MAX_HOURS \
    --resume --ct2
hf upload "$OUT_REPO" /workspace/ft --repo-type model
echo "TRAINING_COMPLETE"
ONSTART_EOF
)

# --- rent, and make sure we always give it back ------------------------------
CREATE_ARGS=(create instance "$OFFER_ID"
    --image "$IMAGE"
    --disk "$DISK"
    --env "-e HF_TOKEN=$HF_TOKEN -e HF_HUB_ENABLE_HF_TRANSFER=1"
    --onstart-cmd "$ONSTART"
    --raw)
if [ "$SPOT" = "1" ]; then
    CREATE_ARGS+=(--bid "$OFFER_DPH")
fi

INSTANCE_ID="$(vastai "${CREATE_ARGS[@]}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["new_contract"])')"
echo "=== rented instance $INSTANCE_ID ==="

destroy() {
    echo ""
    echo "=== destroying instance $INSTANCE_ID ==="
    # retry: leaking a running GPU is the one failure that costs real money
    for _ in 1 2 3 4 5; do
        vastai destroy instance "$INSTANCE_ID" && return 0
        sleep 5
    done
    echo "!! COULD NOT DESTROY $INSTANCE_ID -- destroy it by hand NOW:" >&2
    echo "   vastai destroy instance $INSTANCE_ID" >&2
    return 1
}
trap destroy EXIT INT TERM

# --- watch until the box says it is done, or the budget runs out -------------
# Local deadline is MAX_HOURS plus slack for image pull, model download and the
# result upload, none of which the trainer's own clock covers.
DEADLINE=$(python3 -c "import time; print(int(time.time() + $MAX_HOURS * 3600 + 2400))")
echo "=== watching (Ctrl-C destroys the instance) ==="

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    sleep 60
    LOG="$(vastai logs "$INSTANCE_ID" --tail 40 2>/dev/null || true)"
    printf '%s' "$LOG" | tail -3

    if printf '%s' "$LOG" | grep -q TRAINING_COMPLETE; then
        echo ""
        echo "=== training complete -> https://huggingface.co/$OUT_REPO ==="
        exit 0
    fi
    if printf '%s' "$LOG" | grep -qE "Traceback|CUDA out of memory|Error:"; then
        echo ""
        echo "=== instance reported an error; destroying and stopping ===" >&2
        vastai logs "$INSTANCE_ID" --tail 100 >&2 || true
        exit 1
    fi
done

echo "=== local deadline hit; destroying ===" >&2
exit 1
