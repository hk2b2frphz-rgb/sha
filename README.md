# term2speech (updated 2026-06-13)

専門用語リストから Gemma 4 で発話例文を生成し、Qwen3-TTS で音声合成する。
音声認識 (ASR) のテストデータ作成用。

`annotations.tsv` から Qwen3.6-27B/vLLMで学習文を作り、Qwen3-TTSを経て
Whisperのencoder凍結・decoder fine-tuneとCTranslate2変換まで行うMiltoka向け手順は
[README_whisper_domain_ft.md](README_whisper_domain_ft.md) を参照。

## パイプライン

```
PDF / PPTX / DOCX / 画像 (スキャン文書 OK)
   ↓ scripts/ocr_documents.py        (Qwen3-VL で OCR)  ※テキスト PDF だけなら省略可
テキストファイル群
   ↓ scripts/extract_terms.py        (pypdf + Gemma 4)
terms.txt (1 行 1 用語) ← 手書きのリストでも OK
   ↓ scripts/generate_sentences.py   (Gemma 4)  ← 例文生成と同時に読み仮名も確認
sentences.jsonl  {"id", "term", "sentence", "tts_text"}
   ↓ scripts/synthesize_speech.py    (Qwen3-TTS)  ← tts_text を使って音声合成
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
#    sentences.jsonl には tts_text が含まれており、synthesize_speech.py が自動的に使用する
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
| annotate_readings.py | `--model` | gpt-4o | 読み確認に使う OpenAI モデル |
| | `--interval` | 0.5 | API 呼び出し間隔（秒） |
| generate_sentences.py | `--sentences-per-term` | 3 | 用語あたりの例文数 |
| | `--model` | google/gemma-4-E2B-it | Gemma モデル ID |
| | `--temperature` | 0.8 | 生成の多様性 |
| synthesize_speech.py | `--speaker` | Ono_Anna | Qwen3-TTS プリセット話者 |
| | `--instruct` | (なし) | 話し方のスタイル指示 |
| | `--model` | Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice | TTS モデル ID |

## CPU で作って、学習だけ GPU を借りる

GPU を借りるのは学習の数時間だけにして、それ以外は CPU コンテナで完結させる構成。

| 工程 | 実行場所 | 根拠 |
|---|---|---|
| 学習文の準備 (`prepare_train_from_csv.py`) | **CPU** | 文字列処理のみ |
| TTS 合成 (Kokoro-82M) | **CPU** | 82M と小さく、4 vCPU で実時間より速い (約 1.5 秒/文) |
| manifest 構築・分割 | **CPU** | ファイル処理のみ |
| **学習 (whisper-large-v3-turbo)** | **GPU (Vast.ai)** | ここだけ GPU が要る |
| 推論・評価 (`eval_whisper_hf.py`) | **CPU** | RTF 約 3。件数が多ければ GPU でもよい |

```bash
# 0. 一度だけ: CPU 用 venv (pyproject の cu121 固定とは別に作る)
bash scripts/setup_cpu_env.sh

# 1. CPU だけでデータ生成 (GPU 不使用)
bash scripts/run_cpu_dataprep.sh data/generated_sentences.csv
#    疎通確認なら先頭20件だけ: LIMIT=20 bash scripts/run_cpu_dataprep.sh <csv>

# 2. HF に上げる
source .venv-cpu/bin/activate
hf upload <data-repo> out/whisper_turbo --repo-type dataset \
    --include 'train_manifest.jsonl' --include 'train.jsonl' \
    --include 'dev.jsonl' --include 'tts_data/**'

# 3. 学習だけ GPU を借りる
HF_TOKEN=hf_xxx bash scripts/run_whisper_train_vast.sh <data-repo> <out-repo>

# 4. 出来たモデルを CPU で評価
hf download <out-repo> --local-dir out/ct2
python scripts/eval_whisper_hf.py --manifest out/whisper_turbo/dev.jsonl \
    --out-dir out/eval_ft --model-dir out/ct2 --device cpu --dtype float32
```

学習データは CPU 側で作った絶対パスを含むため、借りたマシン上では
`rebase_manifest_paths.py` が `audio` を実体に貼り替える (ランナーが自動実行)。
`train_manifest.jsonl` が data-repo に無ければランナーは即座に失敗する
— 借りた GPU で TTS を回すのが一番もったいないため。

## Vast.ai (GPU レンタル)

学習だけ [Vast.ai](https://vast.ai/) の従量課金 GPU を使う。

### セットアップ

Claude Code on the web のセッションでは `.claude/hooks/session-start.sh` が
自動で `vastai` CLI を入れ、`VAST_API_KEY` を設定する。手動で入れる場合:

```bash
pip install --user vastai
export PATH="$HOME/.local/bin:$PATH"
vastai set api-key "$VAST_API_KEY"
vastai show user            # 疎通確認 (残高・アカウント情報)
```

`VAST_API_KEY` はリポジトリにコミットせず、claude.ai/code の
environment settings に環境変数として登録する。

### Whisper FT を Vast.ai で回す

`run_whisper_train.pbs` を借りた GPU 上でそのまま実行するラッパ
`scripts/run_whisper_train_vast.sh` を使う。学習の中身は複製していないので、
`FT_MODE` / `FREEZE_ENCODER` / `EPOCHS` / `LR` などは PBS 版と同じ意味。

```bash
export HF_TOKEN=hf_xxx    # データ取得と結果アップロードに使う

# まず価格だけ確認 (課金なし)
DRY_RUN=1 bash scripts/run_whisper_train_vast.sh <hf-data-repo> <hf-out-repo>

# 実行 (既定: full-FT, encoder 凍結)
bash scripts/run_whisper_train_vast.sh me/turbo-tts me/whisper-turbo-ft

# LoRA に切替、上限2時間
FT_MODE=lora MAX_HOURS=2 bash scripts/run_whisper_train_vast.sh me/d me/m
```

第1引数の HF dataset repo が入出力の置き場になる。

| ファイル | 役割 |
|---|---|
| `generated_sentences.csv` | 学習文の入力 (初回に必要) |
| `train_manifest.jsonl`, `tts_data/` | 合成済み音声。初回実行後に自動アップロードされる |

**2 回目以降は TTS を丸ごとスキップする。** `run_whisper_train.pbs` は
`train_manifest.jsonl` があれば step 1 を飛ばす作りなので、合成音声を
HF に置いておけばハイパラ変更だけの再学習が一番重い工程を省ける。

| 環境変数 | 既定値 | 説明 |
|---|---|---|
| `MAX_HOURS` | 4 | ローカル監視側の上限。超過で破棄 |
| `MAX_DPH` | 0.25 | この $/hr を超える提示は借りない (超過時 exit 1) |
| `GPU` | RTX_3090,RTX_4090 | bf16 対応カードのみ |
| `NUM_SHARDS` | 1 | 枚数を増やすと固定作業 (image pull/uv sync/モデルDL) も枚数分課金される |
| `MIXED_PRECISION` | bf16 | PBS 既定の V100 は bf16 非対応なので、そちらは fp16 のまま |
| `SPOT` | 0 | 1 で interruptible 入札 (約 15% 安いが中断あり) |
| `DRY_RUN` | 0 | 1 で提示価格を表示して終了 |

**インスタンスは全ての終了経路で destroy される** — 正常終了・エラー・Ctrl-C・
ローカル締切のいずれでも。放置された GPU が一番の出費になるため
(`$0.11/h` × 一晩 12 時間 ≈ 残高 $10 の 13%)。

### 手動で借りる場合

```bash
vastai search offers 'gpu_name in [RTX_3090,RTX_4090] num_gpus=1 rentable=true' -o 'dph+'
vastai create instance <OFFER_ID> --image pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel --disk 60 --ssh
vastai show instances
vastai destroy instance <INSTANCE_ID>    # 終わったら必ず
```

`vastai stop instance` は停止するだけでディスク課金が残るので、
使い終わったら `destroy` を使う。

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
{"id": "0001", "term": "深層学習", "sentence": "深層学習の研究発表を聴講した。", "wav": "wav/0001.wav", "duration_sec": 3.42, "speaker": "Vivian"}
```
