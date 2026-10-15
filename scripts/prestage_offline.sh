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
# Overrides: BASE_MODEL_FILE, TTS_MODEL, LLM_MODEL, PRONUNCIATION_ASR_REPO,
#            WHISPER_STREAMING_DIR, WHISPER_STREAMING_REF. Set PRESTAGE_LLM=0
#            to skip the large LLM.

set -euo pipefail
cd "${PBS_O_WORKDIR:-$(pwd)}"

BASE_MODEL_FILE="${BASE_MODEL_FILE:-manifest.txt}"
TTS_MODEL="${TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}"
# The text PBS also selects this checkpoint when OFFLINE=1.  Override the same
# LLM_MODEL in both commands if the full BF16 checkpoint is intentionally used.
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3.6-27B-FP8}"
PRONUNCIATION_ASR_REPO="${PRONUNCIATION_ASR_REPO:-mobiuslabsgmbh/faster-whisper-large-v3-turbo}"
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
# Prefer an explicit BASE_MODEL= setting.  Older manifests contain a bare model
# path on the first non-comment line, which remains supported as a fallback.
BASE_MODEL="$(sed -n 's/^[[:space:]]*BASE_MODEL[[:space:]]*=[[:space:]]*//p' "$BASE_MODEL_FILE" | head -n1 | tr -d '[:space:]')"
if [[ -z "$BASE_MODEL" ]]; then
    BASE_MODEL="$(grep -v '^[[:space:]]*#' "$BASE_MODEL_FILE" | grep -v '^[[:space:]]*$' | grep -v '=' | head -n1 | tr -d '[:space:]')"
fi
echo "base model: ${BASE_MODEL:-<empty>}"
echo "tts model:  $TTS_MODEL"
echo "pronunciation ASR: $PRONUNCIATION_ASR_REPO"
if [[ "${PRESTAGE_LLM:-1}" == "1" ]]; then
    echo "llm model:  $LLM_MODEL"
else
    LLM_MODEL=""
    echo "llm model:  <skipped>"
fi
BASE_MODEL="$BASE_MODEL" TTS_MODEL="$TTS_MODEL" LLM_MODEL="$LLM_MODEL" \
PRONUNCIATION_ASR_REPO="$PRONUNCIATION_ASR_REPO" uv run python - <<'PY'
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
warm(os.environ.get("LLM_MODEL"))
warm(os.environ.get("PRONUNCIATION_ASR_REPO"))
print("[prestage] HF cache dir:", os.environ.get("HF_HOME") or "~/.cache/huggingface (default)")
PY

# Kokoro runs in an isolated uv env (deps conflict with the project). Build that
# env from cache-able wheels and fetch its unidic dictionary + the Kokoro model,
# so an OFFLINE training run can reuse them. Skip with PRESTAGE_KOKORO=0.
if [[ "${PRESTAGE_KOKORO:-1}" == "1" ]]; then
    echo "===== prestage 3b/3: Kokoro isolated env ====="
    TORCH_VERSION="$(uv run python -c 'import torch; print(torch.__version__)')"
    TORCHAUDIO_VERSION="$(uv run python -c 'import torchaudio; print(torchaudio.__version__)')"
    KOKORO_MODEL="${KOKORO_MODEL:-hexgrad/Kokoro-82M}"
    KOKORO_UV=(
        uv run --isolated --no-project
        --index "pytorch-cu121=https://download.pytorch.org/whl/cu121"
        --with "kokoro>=0.9.4"
        --with "misaki[ja]"
        --with unidic
        --with pyopenjtalk
        --with soundfile
        --with numpy
        --with "torch==$TORCH_VERSION"
        --with "torchaudio==$TORCHAUDIO_VERSION"
    )
    "${KOKORO_UV[@]}" python -m unidic download
    KOKORO_MODEL="$KOKORO_MODEL" "${KOKORO_UV[@]}" python -c \
        "import os; from huggingface_hub import snapshot_download; print('[prestage] cached', snapshot_download(os.environ['KOKORO_MODEL']))"
fi

# The auto-research loop asks a local Gemma for the next configuration to try.
# That runs in the separate gemma_runtime uv env, so warm it (and the model)
# here too. Skip with PRESTAGE_GEMMA=0; the loop still runs without it, falling
# back to evolutionary search.
if [[ "${PRESTAGE_GEMMA:-1}" == "1" ]]; then
    echo "===== prestage 3c/3: gemma_runtime env + proposer model ====="
    GEMMA_MODEL="${GEMMA_MODEL:-google/gemma-4-E4B-it}"
    echo "gemma model: $GEMMA_MODEL"
    uv sync --project gemma_runtime
    GEMMA_MODEL="$GEMMA_MODEL" uv run --project gemma_runtime python - <<'PY'
import os
from huggingface_hub import snapshot_download

repo = os.environ["GEMMA_MODEL"]
if os.path.isdir(repo):
    print(f"[prestage] {repo} is a local dir; skip download")
else:
    print(f"[prestage] downloading {repo} ...")
    print("[prestage] cached", snapshot_download(repo))
PY
fi

echo "===== prestage done ====="
echo "Next, on the GPU node (no network):"
echo "  qsub -v OFFLINE=1 scripts/run_generate_training_text.pbs"
echo "  qsub -v OFFLINE=1 scripts/run_whisper_train.pbs"
echo "  qsub -v OFFLINE=1 scripts/run_autoresearch.pbs"
echo "  (add other overrides after a comma, e.g. -v \"OFFLINE=1,EPOCHS=8\")"
