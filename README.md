# term2speech

専門用語リストから Gemma 4 で発話例文を生成し、Qwen3-TTS で音声合成する。
音声認識 (ASR) のテストデータ作成用。

## パイプライン

```
PDF / PPTX / DOCX / 画像 (スキャン文書 OK)
   ↓ scripts/ocr_documents.py        (Qwen3-VL で OCR)  ※テキスト PDF だけなら省略可
テキストファイル群
   ↓ scripts/extract_terms.py        (pypdf + Gemma 4)
terms.txt (1 行 1 用語) ← 手書きのリストでも OK
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
# 0. (推奨) 文書 → OCR → 専門用語リストまで一括実行
#    PBS クラスタの場合:
mkdir -p logs
qsub scripts/run_extract.pbs                              # docs/ -> out/terms.txt
qsub -v DOCS_DIR=mydocs,MIN_COUNT=2 scripts/run_extract.pbs   # 入力やオプション変更
#    直接実行の場合:
bash scripts/run_extract.sh docs/          # -> out/terms.txt
# テキスト PDF のみなら: SKIP_OCR=1 bash scripts/run_extract.sh docs/
# ノイズ削減: MIN_COUNT=2 MAX_TERMS=100 bash scripts/run_extract.sh docs/

# --- 以下は個別実行する場合 ---

# 0a. (任意) PDF/PPTX/DOCX/画像 → テキスト (Qwen3-VL で OCR)
#     スキャン PDF やパワポを使う場合のみ。PPTX/DOCX には LibreOffice が必要。
uv run --project gemma_runtime python scripts/ocr_documents.py \
    --docs docs/ \
    --out-dir out/text

# 0b. (任意) 文書 → 専門用語リスト (Gemma 4)
#     テキスト PDF なら docs/ を直接渡してよい (OCR 不要)
uv run --project gemma_runtime python scripts/extract_terms.py \
    --inputs out/text \
    --out out/terms.txt

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
(男性: Uncle_Fu, Dylan, Eric, Ryan, Aiden / 日本語ネイティブ: Ono_Anna のみ)

## 主なオプション

| スクリプト | オプション | 既定値 | 説明 |
|---|---|---|---|
| ocr_documents.py | `--model` | Qwen/Qwen3-VL-8B-Instruct | OCR 用 VLM の モデル ID |
| | `--scale` | 2.0 | PDF レンダリング倍率 (文字が潰れるなら上げる) |
| extract_terms.py | `--chunk-chars` | 3000 | Gemma に渡すチャンクの文字数 |
| | `--min-count` | 1 | この回数以上のチャンクに出た用語のみ採用 |
| | `--max-terms` | 0 (無制限) | 出力する用語数の上限 |
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
