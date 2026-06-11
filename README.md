# term2speech

専門用語リストから Gemma 4 で発話例文を生成し、Qwen3-TTS で音声合成する。
音声認識 (ASR) のテストデータ作成用。

## パイプライン

```
terms.txt (1 行 1 用語)
   ↓ scripts/generate_sentences.py   (Gemma 4)
sentences.jsonl  {"id", "term", "sentence"}
   ↓ scripts/synthesize_speech.py    (Qwen3-TTS)
out/audio/wav/*.wav + manifest.jsonl (正解テキスト付き)
```

manifest.jsonl が ASR テストの正解 (リファレンス) になる。

## セットアップ

```bash
uv sync                           # Qwen3-TTS 用
uv sync --project gemma_runtime   # Gemma 4 用 (transformers 5.x)
```

**要件**: Python 3.11+, NVIDIA GPU, CUDA 対応 PyTorch

## 使い方

```bash
# 1. 用語 → 発話例文 (Gemma 4)
uv run --project gemma_runtime python scripts/generate_sentences.py \
    --terms terms_example.txt \
    --out out/sentences.jsonl \
    --sentences-per-term 3

# 2. 例文 → 音声 (Qwen3-TTS)
uv run python scripts/synthesize_speech.py \
    --sentences out/sentences.jsonl \
    --out-dir out/audio \
    --speaker Vivian
```

## 主なオプション

| スクリプト | オプション | 既定値 | 説明 |
|---|---|---|---|
| generate_sentences.py | `--sentences-per-term` | 3 | 用語あたりの例文数 |
| | `--model` | google/gemma-4-E2B-it | Gemma モデル ID |
| | `--temperature` | 0.8 | 生成の多様性 |
| synthesize_speech.py | `--speaker` | Vivian | Qwen3-TTS プリセット話者 |
| | `--instruct` | (なし) | 話し方のスタイル指示 |
| | `--model` | Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice | TTS モデル ID |

## トラブルシューティング: libcudart.so.13 が開けない (V100)

原因の見立て: CUDA 13 ビルドのパッケージが環境に混入している。
プロジェクトは CUDA 12.1 (cu121) の PyTorch を指定しており、
CUDA 13 は V100 (Volta) をサポートしないため、全パッケージを CUDA 12 系に揃える必要がある。

以下を GPU マシンで実行して、出力を確認する:

```bash
cd term2speech

# 1. torch 自体は生きているか
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

# 2. どのパッケージが libcudart.so.13 を要求しているか特定
uv run python -X importtime -c "from qwen_tts import Qwen3TTSModel" 2>&1 | grep -i -B2 cudart

# 3. CUDA 13 系の nvidia パッケージが混入していないか
uv pip list | grep -i -E "nvidia|cu13|torch"

# (gemma_runtime 側で同じ症状が出る場合)
uv run --project gemma_runtime python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
uv pip list --project gemma_runtime | grep -i -E "nvidia|cu13|torch"
```

出力結果をもとに pyproject.toml を修正する
(例: `torch>=2.1.0,<2.6` で上限固定、cu13 系パッケージを cu12 系へ差し替え)。

### V100 での注意

V100 は bfloat16 をハードウェアサポートしないため、音声合成では `--dtype float16` を明示する:

```bash
uv run python scripts/synthesize_speech.py \
    --sentences out/sentences.jsonl --out-dir out/audio \
    --dtype float16
```

## 出力フォーマット

`out/audio/manifest.jsonl` (1 行 1 音声):

```json
{"id": "0001", "term": "心筋梗塞", "sentence": "祖父が心筋梗塞で入院したと連絡があった。", "wav": "wav/0001.wav", "duration_sec": 3.42, "speaker": "Vivian"}
```
