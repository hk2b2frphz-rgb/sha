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

# Kokoro runs in an isolated uv env (deps conflict with the project) and lazily
# downloads several things at first synthesis: the unidic dictionary, the
# pyopenjtalk open_jtalk dict, misaki[ja] data, and the Kokoro model. Warm ALL of
# them here by building the env and running one real synthesis, so an OFFLINE
# training run needs no network. Uses the SAME --with set as run_whisper_train.pbs
# so uv reuses the cached env (and its downloaded dictionaries). Skip with
# PRESTAGE_KOKORO=0.
if [[ "${PRESTAGE_KOKORO:-1}" == "1" ]]; then
    echo "===== prestage 3b/3: Kokoro isolated env + dictionaries ====="
    TORCH_VERSION="$(uv run python -c 'import torch; print(torch.__version__)')"
    TORCHAUDIO_VERSION="$(uv run python -c 'import torchaudio; print(torchaudio.__version__)')"
    KOKORO_MODEL="${KOKORO_MODEL:-hexgrad/Kokoro-82M}"
    KOKORO_VOICE="${KOKORO_VOICE:-jf_alpha}"
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
    echo "[prestage] fetching unidic dictionary..."
    "${KOKORO_UV[@]}" python -m unidic download
    echo "[prestage] warming Kokoro with one synthesis (pulls model + pyopenjtalk/misaki dict)..."
    warm_dir="out/whisper_turbo/_kokoro_warm"
    mkdir -p "$warm_dir"
    # One Japanese warm sentence (needed to trigger the JP dictionary downloads).
    # This .sh runs on the login node via `bash`, not qsub, so a UTF-8 literal is
    # fine here (the no-Japanese rule is only for .pbs jobs).
    "${KOKORO_UV[@]}" python -c "import json,sys; open(sys.argv[1],'w',encoding='utf-8').write(json.dumps({'id':'warm_0001','sentence':'これはテストです'},ensure_ascii=False)+'\n')" "$warm_dir/sentences.jsonl"
    # CPU on the login node (no GPU there); this just triggers the downloads.
    "${KOKORO_UV[@]}" python scripts/synthesize_speech_kokoro.py \
        --sentences "$warm_dir/sentences.jsonl" \
        --out-dir "$warm_dir" \
        --model-id "$KOKORO_MODEL" \
        --voice "$KOKORO_VOICE" \
        --device cpu
    echo "[prestage] Kokoro warm synth OK -> $warm_dir/wav/warm_0001.wav"
fi

echo "===== prestage done ====="
echo "Next, on the GPU node (no network):"
echo "  qsub -v OFFLINE=1 scripts/run_whisper_train.pbs"
echo "  (add other overrides after a comma, e.g. -v \"OFFLINE=1,EPOCHS=8\")"
