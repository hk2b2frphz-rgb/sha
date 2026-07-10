#!/usr/bin/env bash
# Prestage for OFFLINE training on network-isolated GPU nodes (e.g. res=middle2).
#
# Run this on a node WITH internet (the login node, or a networked compute node)
# so the later `OFFLINE=1 qsub scripts/run_whisper_train.pbs` on middle2 needs no
# network at all. It populates, on the shared filesystem that middle2 can read:
#   - the uv .venv (dependencies)
#   - the HuggingFace cache: Qwen3-TTS model + whisper-large-v3-turbo base
#   - vendor/whisper_streaming (used by eval)
#
# Run on the login node:
#   PROXY_URL=http://user:pass%40@host:port bash scripts/prestage_offline.sh
#
# Overrides: BASE_MODEL_FILE, TTS_MODEL, WHISPER_STREAMING_DIR, WHISPER_STREAMING_REF

set -euo pipefail
cd "${PBS_O_WORKDIR:-$(pwd)}"

BASE_MODEL_FILE="${BASE_MODEL_FILE:-manifest.txt}"
TTS_MODEL="${TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}"
WHISPER_STREAMING_DIR="${WHISPER_STREAMING_DIR:-vendor/whisper_streaming}"
WHISPER_STREAMING_REF="${WHISPER_STREAMING_REF:-main}"

# Proxy is needed to reach the internet from the login node.
if [[ -f "$PWD/scripts/setup_proxy.sh" ]]; then
    # shellcheck source=/dev/null
    source "$PWD/scripts/setup_proxy.sh"
fi

command -v uv >/dev/null 2>&1 || { echo "ERROR: uv is not on PATH." >&2; exit 1; }
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

echo "===== prestage 1/3: uv sync ====="
uv sync

echo "===== prestage 2/3: clone whisper_streaming ====="
mkdir -p vendor
if [[ ! -d "$WHISPER_STREAMING_DIR/.git" ]]; then
    git clone https://github.com/ufal/whisper_streaming.git "$WHISPER_STREAMING_DIR"
fi
git -C "$WHISPER_STREAMING_DIR" fetch --depth 1 origin "$WHISPER_STREAMING_REF" || true
git -C "$WHISPER_STREAMING_DIR" checkout "$WHISPER_STREAMING_REF" || true

echo "===== prestage 3/3: warm HuggingFace cache ====="
# base whisper model path/id from manifest.txt (first non-comment line)
BASE_MODEL="$(grep -v '^[[:space:]]*#' "$BASE_MODEL_FILE" | grep -v '^[[:space:]]*$' | head -n1 | tr -d '[:space:]')"
echo "base model: ${BASE_MODEL:-<empty>}"
echo "tts model:  $TTS_MODEL"
BASE_MODEL="$BASE_MODEL" TTS_MODEL="$TTS_MODEL" uv run python - <<'PY'
import os
from huggingface_hub import snapshot_download

def warm(repo):
    if not repo:
        return
    if os.path.isdir(repo):
        print(f"[prestage] {repo} is a local dir; skip download")
        return
    print(f"[prestage] downloading {repo} ...")
    path = snapshot_download(repo)
    print(f"[prestage] cached {repo} -> {path}")

warm(os.environ.get("BASE_MODEL"))
warm(os.environ.get("TTS_MODEL"))
print("[prestage] HF cache dir:", os.environ.get("HF_HOME") or "~/.cache/huggingface (default)")
PY

echo "===== prestage done ====="
echo "Next, on the GPU node (no network):"
echo "  qsub -v OFFLINE=1 scripts/run_whisper_train.pbs"
echo "  (add other overrides after a comma, e.g. -v \"OFFLINE=1,EPOCHS=8\")"
