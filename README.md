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
    --speaker Ono_Anna
```

プリセット話者: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee
(日本語は Ono_Anna が自然)

## 主なオプション

| スクリプト | オプション | 既定値 | 説明 |
|---|---|---|---|
| generate_sentences.py | `--sentences-per-term` | 3 | 用語あたりの例文数 |
| | `--model` | google/gemma-4-E2B-it | Gemma モデル ID |
| | `--temperature` | 0.8 | 生成の多様性 |
| synthesize_speech.py | `--speaker` | Ono_Anna | Qwen3-TTS プリセット話者 |
| | `--instruct` | (なし) | 話し方のスタイル指示 |
| | `--model` | Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice | TTS モデル ID |

## 環境診断 (トラブルシューティング)

CUDA / torch / qwen-tts まわりのエラー (libcudart が開けない等) が出たら、
診断スクリプトを実行する。ステップごとに OK/NG と診断まとめが表示される。

```bash
python3 scripts/check_env.py
```

NG がある場合は全出力をコピーして共有する。

注意: V100 は bfloat16 非対応のため、synthesize_speech.py には `--dtype float16` を付けること。

## 出力フォーマット

`out/audio/manifest.jsonl` (1 行 1 音声):

```json
{"id": "0001", "term": "心筋梗塞", "sentence": "祖父が心筋梗塞で入院したと連絡があった。", "wav": "wav/0001.wav", "duration_sec": 3.42, "speaker": "Vivian"}
```
