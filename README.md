# term2speech (updated 2026-06-13)

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
   ↓ scripts/finetune_whisper.py     (Whisper large-v3 を LoRA で追加学習)
out/whisper-ft/ct2/ (faster-whisper でそのまま読める)
```

manifest.jsonl が ASR テストの正解 (リファレンス) になる。
同じものを教師データとして Whisper の fine tuning にも使える。

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

## Vast.ai (GPU レンタル)

このリポジトリは GPU を前提とするため、手元に GPU がない場合は
[Vast.ai](https://vast.ai/) の従量課金 GPU を使う。

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

### インスタンスを借りる

Whisper large-v3 の LoRA fine tuning なら VRAM 24GB あれば足りる。
**bfloat16 対応の RTX 3090 (Ampere) / 4090 (Ada) を選ぶこと。**
T4 (Turing) と V100 は bf16 非対応で、fp16 学習は勾配が NaN に飛びやすい。

```bash
# 安い順に検索 (2026-08 時点で RTX 3090 が $0.11/hr 前後)
vastai search offers 'gpu_name in [RTX_3090,RTX_4090] num_gpus=1 disk_space>=100 rentable=true' -o 'dph+'

# 借りる (ID は上の検索結果から)
vastai create instance <OFFER_ID> \
    --image pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel \
    --disk 100 --ssh

vastai show instances       # 状態と SSH 接続先を確認
vastai ssh-url <INSTANCE_ID>
```

**課金はインスタンスが起動している間ずっと発生する。終わったら必ず破棄する:**

```bash
vastai destroy instance <INSTANCE_ID>
```

`vastai stop instance` は停止するだけでディスク課金が残るので、
使い終わったら `destroy` を使う。

## Whisper の fine tuning

`manifest.jsonl` を教師データに Whisper large-v3 を LoRA で追加学習し、
faster-whisper (CTranslate2) 形式まで書き出す。

```bash
# GPU のあるマシンで直接
python scripts/finetune_whisper.py \
    --manifest out/audio/manifest.jsonl \
    --out-dir out/whisper-ft \
    --max-hours 3 --ct2
```

出力は `out/whisper-ft/` 以下:

| パス | 中身 |
|---|---|
| `adapter/` | LoRA アダプタのみ (数十 MB) |
| `merged/` | ベースモデルに統合済みの HF モデル |
| `ct2/` | faster-whisper でそのまま読める形式 (`--ct2` 指定時) |

### GPU 時間を削るための既定値

借りた GPU で回す前提なので、既定値は精度より請求額を優先している。

| 設定 | 理由 |
|---|---|
| **エンコーダ凍結** | 用語適応はデコーダ側が効く。逆伝播を省いて 1 step 約 35% 短縮 |
| **LoRA (decoder q/v)** | 学習対象は全パラメータの 1% 未満 |
| **bf16・量子化なし** | large-v3 は約 3 GB。24 GB カードで 8bit 化しても脱量子化の分だけ遅くなる |
| **log-mel を都度計算** | 特徴量キャッシュ数十 GB のディスク課金を回避。DataLoader が GPU と重ねて処理 |
| `--max-hours` | 壁時計で強制終了。請求額の上限を保証する |
| `--resume` | spot インスタンスが飛ばされても再開できる |

精度を優先する場合は `--train-encoder` を付ける (その分 GPU 時間が伸びる)。

### Vast.ai で回す (借りて・学習して・自動で返す)

```bash
export HF_TOKEN=hf_xxx    # データ取得と結果アップロードに使う

# まず価格だけ確認 (課金なし)
DRY_RUN=1 bash scripts/vast_train.sh <hf-dataset-repo> <hf-output-repo>

# 実行
MAX_HOURS=3 bash scripts/vast_train.sh myname/term2speech-audio myname/whisper-ja-terms
```

**インスタンスは全ての終了経路で destroy される** — 正常終了・学習失敗・Ctrl-C・
ローカルのタイムアウトのいずれでも。放置された GPU が一番の出費になるため
(`$0.11/h` × 一晩 12 時間 ≈ 残高 $10 の 13%)。

| 環境変数 | 既定値 | 説明 |
|---|---|---|
| `MAX_HOURS` | 3 | 学習側とローカル監視側の両方に効く上限 |
| `MAX_DPH` | 0.20 | この $/hr を超える提示は借りない (超過時は exit 1) |
| `GPU` | RTX_3090,RTX_4090 | bf16 対応カードのみ。T4/V100 は除外 |
| `SPOT` | 0 | 1 で interruptible 入札 (約 15% 安いが中断あり) |
| `DISK` | 40 | GB。large-v3 と出力には十分 |
| `DRY_RUN` | 0 | 1 で提示価格を表示して終了 |

学習結果は HF Hub にアップロードされてからインスタンスが破棄される。
2026-08 時点の実測で RTX 3090 が **$0.099/hr**、`MAX_HOURS=3` なら最悪 **$0.30**。

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
