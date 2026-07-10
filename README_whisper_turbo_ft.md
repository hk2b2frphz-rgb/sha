# whisper-large-v3-turbo FT パイプライン (generated_sentences.csv 起点)

`generated_sentences.csv` の専門用語だけをひらがな読みにした発話を TTS で合成し、
その音声を使って whisper-large-v3-turbo を LoRA fine-tune、CT2 へ変換して
whisper-streaming で評価する、ワンパスの手順。

- **学習ターゲット**: 元の `Sentance`(漢字のまま)。TTSはひらがな読みで発話し、whisperには
  「正しい読みの音声 → 漢字表記」を学習させる。
- **TTS**: 既定は **Kokoro-82M**(高速)。`TTS_BACKEND=qwen3` で Qwen3-TTS に切替可。
- **FT方式**: LoRA/PEFT。学習後 base にマージ→CT2変換。
- **eval対象**: 共有された test wav 群のみ。参照テキストがあれば CER/WER も算出。
- **既定リソース**: 学習=`xan_s` / `res=middle`(**A100×2**)、評価=`res=small`。

## 0. 事前準備

- `data/generated_sentences.csv` を配置。列: `Number,Word,Reading,Sentance,AI_Reading,Confirmed_Reading`
- `manifest.txt` の先頭非コメント行を、HPC上の whisper-large-v3-turbo 重みパスに書き換える
  (既定は `openai/whisper-large-v3-turbo`)。

## ノード / TTSバックエンド

既定は **A100×2 (`xan_s` / `res=middle`) + Kokoro-TTS**。`run_whisper_train.pbs` に
`#PBS -q xan_s` / `#PBS -l select=1:res=middle` が入っているので、そのまま投げるとA100で動く。

```bash
# 既定 (A100×2, Kokoro, オンライン)
qsub -v "PROXY_URL=http://user:pass%40@host:port" scripts/run_whisper_train.pbs

# Qwen3-TTS に切替
qsub -v "TTS_BACKEND=qwen3,PROXY_URL=http://user:pass%40@host:port" scripts/run_whisper_train.pbs
```

Kokoro は依存が本体と衝突する(misaki[ja]がfull unidicを要求 vs 本体のunidic-lite)ため、
**隔離env** (`uv run --isolated --with kokoro ... --with 'misaki[ja]'`) で動かす。ジョブが
自動でその env を作り `python -m unidic download` する(初回は少し時間がかかる)。

**V100×4 (`res=middle2`) で回す場合はネット遮断なのでオフライン実行**:

```bash
# 1) ネット可能ノードで事前DL (.venv + HFキャッシュ + Kokoro隔離env + vendor)
PROXY_URL=http://user:pass%40@host:port bash scripts/prestage_offline.sh
# 2) V100×4でオフライン実行
qsub -q xvn_s -l select=1:res=middle2 -v "OFFLINE=1,NUM_SHARDS=4" scripts/run_whisper_train.pbs
```

GPU枚数は `NUM_SHARDS` から自動導出(既定2)。`CUDA_VISIBLE_DEVICES` を渡す必要はない。

**進捗表示**: メインのジョブログに `[tts-progress] 済/総 (％) elapsed=...` を
`PROGRESS_EVERY`秒(既定30)ごとに出力。1文ごとの詳細(ETA付き)は
`out/whisper_turbo/tts_data/shard_*.log`。TTSは自己回帰生成なので単体高速化は限定的で、
実質は「Kokoro(既定)」「A100」「GPUを増やす(NUM_SHARDS)」が効く。

`OFFLINE=1` で `UV_OFFLINE`/`HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` を立て、`uv sync` は
キャッシュのみ、モデルDLもキャッシュから読む。ノードがネットに繋がる場合は不要。

疎通確認(学習と同じノード種別で): `qsub -l select=1:res=middle2 -v "PROXY_URL=..." scripts/run_net_check.pbs`

## 1. 学習 (データ生成 + LoRA FT + CT2変換)

スクリプトには `#PBS -V` があるので、上書き変数は **`qsub -v VAR=値`** で明示的に渡す
(シェル前置き `VAR=値 qsub` ではなく `-v`。複数はカンマ区切り)。

```bash
qsub scripts/run_whisper_train.pbs
# ハイパラ上書き:
qsub -v "CSV=data/generated_sentences.csv,EPOCHS=8,LORA_R=64" scripts/run_whisper_train.pbs
# プロキシを渡す場合:
qsub -v "PROXY_URL=http://user:pass@proxy.example.com:8080" scripts/run_whisper_train.pbs
```

1. `prepare_train_from_csv.py`: CSV → `tts_input.jsonl` (`Word`→`Reading` 置換で `tts_text`、
   元漢字文を `sentence` に保持)
2. 4GPU 並列で `synthesize_speech.py` を実行して TTS 音声を生成
3. `build_whisper_manifest.py`: 音声パス + 元漢字文 → `train_manifest.jsonl`
4. `train_whisper_lora.py` を `accelerate launch`(4GPU DDP) で実行 → LoRA アダプタ + マージ済みHFモデル
5. `ct2-transformers-converter` で CT2 形式へ変換

**学習データの再利用**: `TRAIN_MANIFEST`(既定 `out/whisper_turbo/train_manifest.jsonl`) が
既にあれば TTS 生成をスキップする。ハイパラだけ変えて再投入すれば TTS を作り直さない。
作り直したいときは `FORCE_REBUILD_DATA=1`。

出力(既定):
- `out/whisper_turbo/train_manifest.jsonl` — 再利用される学習データ
- `out/whisper_turbo/lora/adapter/` — LoRA アダプタ
- `out/whisper_turbo/merged_hf/` — マージ済み HF モデル
- `out/whisper_turbo/ct2/` — **評価に使う CT2 モデル**

## 2. 評価 (共有 test wav を whisper-streaming で推論)

test wav 群を `data/test_wav/` などに置く。参照テキスト(正解)があれば以下いずれかで用意する:

- **位置対応(推奨)**: 1行1正解テキストのみ。wav の順番(自然順ソート: `1.wav, 2.wav, 10.wav`)
  と行番号で対応付ける。キー列は不要。
- キー付きTSV: `key<TAB>text` (`key` は wav ファイル名 or stem)。行にタブがあればこちらと判定。

```bash
qsub -v WAV_DIR=data/test_wav scripts/run_whisper_eval.pbs
# 参照ありで CER/WER も出す場合 (位置対応ファイル):
qsub -v "WAV_DIR=data/test_wav,REFS=data/test_refs.txt" scripts/run_whisper_eval.pbs
# プロキシも一緒に:
qsub -v "PROXY_URL=http://user:pass@proxy.example.com:8080,WAV_DIR=data/test_wav" scripts/run_whisper_eval.pbs
```

`build_test_manifest.py` が `test_manifest.jsonl` を作り、
`evaluate_whisper_streaming.py` が CER/WER/RTF を算出する。

出力(既定): `experiments/whisper_turbo_eval/{predictions.jsonl,summary.json,report.md}`

## 主な環境変数

| 変数 | 既定 | 意味 |
|---|---|---|
| `CSV` | `data/generated_sentences.csv` | 入力CSV |
| `BASE_MODEL_FILE` | `manifest.txt` | ベースモデルパスを記した txt |
| `NUM_SHARDS` | `4` | TTS並列数 / DDPプロセス数 (GPU枚数と一致) |
| `EPOCHS`/`LR`/`BATCH_SIZE` | `5`/`1e-4`/`8` | 学習ハイパラ |
| `LORA_R`/`LORA_ALPHA` | `32`/`64` | LoRA ランク |
| `FORCE_REBUILD_DATA` | `0` | `1`でTTSデータを作り直す |
| `MODEL_DIR` (eval) | `out/whisper_turbo/ct2` | 評価するCT2モデル |
| `WAV_DIR`/`REFS` (eval) | `data/test_wav`/なし | test wav群 / 参照TSV |

## メモ

- LoRA対象は attention の `q/k/v/out_proj`。DDPで unused-param エラーが出る場合は
  `train_whisper_lora.py` の `ddp_find_unused_parameters` を `True` に。
- CT2変換は `merged_hf/` に `tokenizer.json`/`preprocessor_config.json` が保存されている前提
  (train スクリプトが processor を保存する)。
