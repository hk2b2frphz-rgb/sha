#!/usr/bin/env bash
# データ生成・推論・評価を CPU だけで回すための venv を作る。
#
# pyproject.toml は torch を cu121 インデックスに固定している (クラスタ用に正しい)。
# ここはそれを触らず、CPU ビルドの torch を入れた別 venv (.venv-cpu) を用意する。
# 学習だけは GPU を借りて行うので、この env に GPU 用の依存は入れない。
#
# Usage:
#   bash scripts/setup_cpu_env.sh
set -euo pipefail

cd "$(dirname "$0")/.."
VENV="${VENV:-.venv-cpu}"

command -v uv >/dev/null || { echo "ERROR: uv がありません。pip install uv" >&2; exit 1; }

echo "=== $VENV を作成 ==="
uv venv "$VENV" --python 3.11
# shellcheck source=/dev/null
source "$VENV/bin/activate"

# CPU ビルドを先に入れる。後続の解決で CUDA ホイール (約 2.5GB) を掴ませない。
echo "=== CPU 版 torch ==="
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "=== データ生成・評価用の依存 ==="
uv pip install numpy soundfile librosa transformers peft accelerate datasets \
    faster-whisper ctranslate2 fugashi unidic-lite PyYAML pytest

python - <<'PY'
import torch
print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
assert not torch.cuda.is_available(), "CPU 用 env のはずが CUDA を掴んでいます"
import faster_whisper, transformers, librosa  # noqa: F401
print("faster-whisper / transformers / librosa OK")
PY

echo ""
echo "=== 完了 ==="
echo "使い方: source $VENV/bin/activate"
echo "次: bash scripts/run_cpu_dataprep.sh data/generated_sentences.csv"
