#!/usr/bin/env bash
# Submit the next pipeline stage from the tail of a finished PBS job.
#
# PBS `-W depend=afterok:` leaves the dependent job held forever when the
# server cannot report the parent's exit status (job history disabled or
# purged), so each stage submits its successor itself instead.  Set CHAIN=0
# to stop after the current stage.
#
# Usage: chain_next <next .pbs script> <label>   (needs REPO and RUN_ROOT)

chain_next() {
    local script=$1 label=$2 vars out
    local -a cmd

    if [[ "${CHAIN:-1}" != "1" ]]; then
        echo "chain: disabled (CHAIN=${CHAIN:-}); submit $label manually when ready"
        return 0
    fi
    [[ -f "$script" ]] || { echo "chain: next stage script not found: $script" >&2; return 0; }

    vars="REPO=$REPO,RUN_ROOT=$RUN_ROOT,CHAIN=1"
    [[ -n "${MIL:-}" ]] && vars="$vars,MIL=$MIL"
    [[ -n "${PROXY_URL:-}" ]] && vars="$vars,PROXY_URL=$PROXY_URL"

    mkdir -p "$RUN_ROOT/logs"
    cmd=(qsub -v "$vars" -o "$RUN_ROOT/logs/" "$script")

    if ! command -v qsub >/dev/null 2>&1; then
        echo "chain: qsub is unavailable on this node; submit $label manually:" >&2
        printf 'chain:  ' >&2; printf ' %q' "${cmd[@]}" >&2; echo >&2
        return 0
    fi

    if out=$("${cmd[@]}" 2>&1); then
        echo "chain: submitted $label -> $out"
    else
        # The current stage succeeded, so report the failure loudly and keep
        # this job's exit status clean; the run resumes from the same RUN_ROOT.
        echo "chain: FAILED to submit $label" >&2
        echo "$out" >&2
        echo "chain: resubmit manually with:" >&2
        printf 'chain:  ' >&2; printf ' %q' "${cmd[@]}" >&2; echo >&2
    fi
    return 0
}
