#!/usr/bin/env bash
# Submit the text, TTS, and decoder-FT PBS jobs with one shared run directory.
# Users only need to provide REPO, MIL, and PROXY_URL. Advanced settings may
# still be exported before invoking this helper and are forwarded where needed.

set -euo pipefail

: "${REPO:?Set REPO to the absolute term2speech checkout path}"
: "${MIL:?Set MIL to the absolute Miltoka checkout path}"
: "${PROXY_URL:?Set PROXY_URL to the URL-encoded proxy URL}"

[[ "$REPO" == /* ]] || { echo "ERROR: REPO must be an absolute path: $REPO" >&2; exit 1; }
[[ "$MIL" == /* ]] || { echo "ERROR: MIL must be an absolute path: $MIL" >&2; exit 1; }
[[ -d "$REPO" ]] || { echo "ERROR: REPO directory not found: $REPO" >&2; exit 1; }
[[ -d "$MIL" ]] || { echo "ERROR: MIL directory not found: $MIL" >&2; exit 1; }
[[ "$PROXY_URL" != *,* ]] || {
    echo "ERROR: PROXY_URL contains a literal comma; encode it as %2C for qsub -v" >&2
    exit 1
}
command -v qsub >/dev/null 2>&1 || { echo "ERROR: qsub is unavailable" >&2; exit 1; }

VLLM_CMD="$MIL/.venv_vllm_qwen_10000/bin/vllm"
VLLM_NINJA="$MIL/.venv_vllm_qwen_10000/bin/ninja"
VLLM_OMNI_CMD="$MIL/.venv-vllm-omni/bin/vllm"
VLLM_OMNI_PYTHON="$MIL/.venv-vllm-omni/bin/python"
VLLM_OMNI_NINJA="$MIL/.venv-vllm-omni/bin/ninja"
PROJECT_PYTHON="$REPO/.venv/bin/python"

for executable in \
    "$VLLM_CMD" "$VLLM_NINJA" \
    "$VLLM_OMNI_CMD" "$VLLM_OMNI_PYTHON" "$VLLM_OMNI_NINJA" \
    "$PROJECT_PYTHON"
do
    [[ -x "$executable" ]] || { echo "ERROR: required executable not found: $executable" >&2; exit 1; }
done

if [[ -n "${ANNOTATIONS:-}" ]]; then
    annotations="$ANNOTATIONS"
elif [[ -s "$REPO/data/annotations.tsv" ]]; then
    annotations="$REPO/data/annotations.tsv"
elif [[ -s "$REPO/annotations.tsv" ]]; then
    annotations="$REPO/annotations.tsv"
else
    echo "ERROR: annotations.tsv not found; place it at $REPO/data/annotations.tsv" >&2
    exit 1
fi
[[ -s "$annotations" ]] || { echo "ERROR: annotations file is missing or empty: $annotations" >&2; exit 1; }

run_root="${RUN_ROOT:-$REPO/out/whisper_domain_ft/run_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$run_root/logs"
common_vars="REPO=$REPO,MIL=$MIL,PROXY_URL=$PROXY_URL,RUN_ROOT=$run_root"

# Each qsub is reported on its own so a rejected stage never dies silently and
# leaves the earlier stages running with no successor.
submit() {
    local label=$1 script=$2 extra=$3 depend=$4 out
    local -a cmd=(qsub -o "$run_root/logs/")
    [[ -n "$depend" ]] && cmd+=(-W "depend=afterok:$depend")
    cmd+=(-v "$common_vars${extra:+,$extra}" "$script")
    if ! out=$("${cmd[@]}" 2>&1); then
        echo "ERROR: failed to submit $label" >&2
        echo "$out" >&2
        echo "resubmit with:" >&2
        printf ' ' >&2; printf ' %q' "${cmd[@]}" >&2; echo >&2
        return 1
    fi
    printf '%s' "$out"
}

# CHAIN=1 (default): stage 1 submits stage 2 when it finishes, and stage 2
# submits stage 3.  Held `depend=afterok` jobs never recover when the PBS
# server cannot report the parent's exit status, so chaining is the default.
# CHAIN=0 restores the original three-job dependency submission.
if [[ "${CHAIN:-1}" == "1" ]]; then
    text_job=$(submit "stage 1 (term_text)" "$REPO/scripts/run_generate_training_text.pbs" \
        "ANNOTATIONS=$annotations,CHAIN=1" "")
    printf 'run_root=%s\ntext=%s\nlogs=%s\n' "$run_root" "$text_job" "$run_root/logs"
    echo 'stages 2 and 3 are submitted by the preceding job when it succeeds'
    exit 0
fi

text_job=$(submit "stage 1 (term_text)" "$REPO/scripts/run_generate_training_text.pbs" \
    "ANNOTATIONS=$annotations,CHAIN=0" "")
tts_job=$(submit "stage 2 (term_tts)" "$REPO/scripts/run_synthesize_training_audio.pbs" \
    "CHAIN=0" "$text_job")
train_job=$(submit "stage 3 (term_ft)" "$REPO/scripts/run_whisper_decoder_train.pbs" \
    "CHAIN=0" "$tts_job")

printf 'run_root=%s\ntext=%s\ntts=%s\ntrain=%s\nlogs=%s\n' \
    "$run_root" "$text_job" "$tts_job" "$train_job" "$run_root/logs"
