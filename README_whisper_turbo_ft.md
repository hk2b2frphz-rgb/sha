# whisper-large-v3-turbo FT パイプライン (generated_sentences.csv 起点)

`generated_sentences.csv` の専門用語だけをひらがな読みにした発話を TTS で合成し、
その音声を使って whisper-large-v3-turbo を LoRA fine-tune、CT2 へ変換して
whisper-streaming で評価する、ワンパスの手順。

- **学習ターゲット**: 元の `Sentance`(漢字のまま)。TTSはひらがな読みで発話し、whisperには
  「正しい読みの音声 → 漢字表記」を学習させる。
- **TTS**: 既定は **Kokoro-82M**(高速)。`TTS_BACKEND=qwen3` で Qwen3-TTS に切替可。
- **FT方式**: 既定は **full-FT (encoder 凍結 / decoder のみ学習)**。`FT_MODE=lora` で従来のLoRA。
- **eval対象**: 共有された test wav 群のみ。参照テキストがあれば CER/WER も算出。
- **既定リソース**: 学習=`xan_s` / `res=middle`(**A100×2**)、評価=`res=small`。

## 0. 事前準備

- `data/generated_sentences.csv` を配置。列: `Number,Word,Reading,Sentance,AI_Reading,Confirmed_Reading`
- `manifest.txt` の先頭非コメント行を、HPC上の whisper-large-v3-turbo 重みパスに書き換える
  (既定は `openai/whisper-large-v3-turbo`)。

## ノード / TTSバックエンド

既定は **V100×4 (`xvn_s` / `res=middle2`) + Kokoro-TTS**。`run_whisper_train.pbs` に
`#PBS -q xvn_s` / `#PBS -l select=1:res=middle2` が入っている。

```bash
# 既定 (V100×4, Kokoro)
qsub scripts/run_whisper_train.pbs

# Qwen3-TTS に切替
qsub -v TTS_BACKEND=qwen3 scripts/run_whisper_train.pbs

# 音声を作成済みなら、1枚GPUで再学習 (TTSは再利用でスキップ)
qsub -l select=1:res=small -v NUM_SHARDS=1 scripts/run_whisper_train.pbs

# A100×2 (TTSが速い) を使う場合
qsub -q xan_s -l select=1:res=middle -v NUM_SHARDS=2 scripts/run_whisper_train.pbs
```

Kokoro は依存が本体と衝突する(misaki[ja]がfull unidicを要求 vs 本体のunidic-lite)ため、
**隔離env** (`uv run --isolated --with kokoro ... --with 'misaki[ja]'`) で動かす。ジョブが
自動でその env を作り `python -m unidic download` する(初回は少し時間がかかる)。

GPU枚数は `NUM_SHARDS` から自動導出(既定4)。`CUDA_VISIBLE_DEVICES` を渡す必要はない。
`res=middle2` がネット遮断の場合は `prestage_offline.sh`(ネット可能ノード) → `OFFLINE=1`。

**進捗表示**: メインのジョブログに `[tts-progress] 済/総 (％) elapsed=...` を
`PROGRESS_EVERY`秒(既定30)ごとに出力。1文ごとの詳細(ETA付き)は
`out/whisper_turbo/tts_data/shard_*.log`。TTSは自己回帰生成なので単体高速化は限定的で、
実質は「Kokoro(既定)」「A100」「GPUを増やす(NUM_SHARDS)」が効く。

`OFFLINE=1` で `UV_OFFLINE`/`HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` を立て、`uv sync` は
キャッシュのみ、モデルDLもキャッシュから読む。ノードがネットに繋がる場合は不要。

疎通確認(学習と同じノード種別で): `qsub -l select=1:res=middle2 -v "PROXY_URL=..." scripts/run_net_check.pbs`

## FT方式 (full / LoRA)

既定は **full-FT + encoder 凍結**(decoder だけ学習)。学習データが TTS 合成音声なので、
encoder まで動かすと合成音の音響特性に過適合して実音声で崩れる。凍結すれば
「音の聞き取りは事前学習のまま、表記だけ専門用語に寄せる」という本来の狙いに合う。
文献でも、合成音のみで適応する場合は encoder 凍結 → decoder のみ更新が定石
([arXiv:2501.12501](https://arxiv.org/pdf/2501.12501)、[arXiv:2206.13240](https://arxiv.org/pdf/2206.13240))。

```bash
# 既定: full-FT, encoder 凍結, lr=1e-5
qsub scripts/run_whisper_train.pbs
# 従来の LoRA に戻す (lr 既定 1e-4)
qsub -v FT_MODE=lora scripts/run_whisper_train.pbs
# encoder も含めて全層学習 (TTSのみのデータでは非推奨)
qsub -v FREEZE_ENCODER=0 scripts/run_whisper_train.pbs
```

**学習率**: full-FT は LoRA の 1/10 が既定 (`LR` 未指定なら full=`1e-5` / lora=`1e-4`)。
Whisper 系の目安は「事前学習の 1/40」で large 系は 5e-6〜1e-5
([HF fine-tune-whisper](https://huggingface.co/blog/fine-tune-whisper)、
[whisper-finetune](https://github.com/vasistalodagala/whisper-finetune))。
LoRA の `1e-4` をそのまま full-FT に使うと壊れるので、`LR` を上書きするときも
`3e-5` 程度までに留める。

出力先はモードで変わる (`WORK_ROOT=out/whisper_turbo`):

| | 中間出力 | CT2 に渡すモデル |
|---|---|---|
| `FT_MODE=full` | `out/whisper_turbo/full/` (trainer, `train_progress.json`) | `out/whisper_turbo/merged_hf/` (学習済み重みを直接保存) |
| `FT_MODE=lora` | `out/whisper_turbo/lora/adapter/` | `out/whisper_turbo/merged_hf/` (base にマージ) |

full-FT の出力はそれ自体が完成したモデルなので、3GB級を二重に置かないよう
`--merge-dir` に直接保存する (CT2 変換ステップは両モードで同じ)。
HF での単体評価は `--adapter` ではなく `--model-dir` を使う:

```bash
uv run python scripts/eval_whisper_hf.py --manifest <dev.jsonl> \
    --model-dir out/whisper_turbo/merged_hf --out-dir out/eval_full
```

## 1. 学習 (データ生成 + FT + CT2変換)

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
4. `train_whisper_lora.py` を `accelerate launch`(4GPU DDP) で実行 → `merged_hf/` に HFモデル
   (full: 学習済み重みをそのまま / lora: アダプタを base にマージ)
5. `ct2-transformers-converter` で CT2 形式へ変換

**学習データの再利用**: `TRAIN_MANIFEST`(既定 `out/whisper_turbo/train_manifest.jsonl`) が
既にあれば TTS 生成をスキップする。ハイパラだけ変えて再投入すれば TTS を作り直さない。
作り直したいときは `FORCE_REBUILD_DATA=1`。

出力(既定):
- `out/whisper_turbo/train_manifest.jsonl` — 再利用される学習データ
- `out/whisper_turbo/full/` (または `lora/adapter/`) — 学習ログ / アダプタ
- `out/whisper_turbo/merged_hf/` — CT2 に渡す HF モデル
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
| `FT_MODE` | `full` | `full`(encoder凍結) / `lora` |
| `FREEZE_ENCODER` | `1` | `0` で encoder も学習 (full のみ) |
| `EPOCHS`/`LR`/`BATCH_SIZE` | `5`/`1e-5`(full) or `1e-4`(lora)/`8` | 学習ハイパラ |
| `LORA_R`/`LORA_ALPHA` | `32`/`64` | LoRA ランク (`FT_MODE=lora` のみ) |
| `FORCE_REBUILD_DATA` | `0` | `1`でTTSデータを作り直す |
| `MODEL_DIR` (eval) | `out/whisper_turbo/ct2` | 評価するCT2モデル |
| `WAV_DIR`/`REFS` (eval) | `data/test_wav`/なし | test wav群 / 参照TSV |

## 3. ハイパラを自動で詰める

学習データ (`train_manifest.jsonl`) を 1 度作ってしまえば、`README_autoresearch.md` の
自動改善ループに投げて、CER を目的関数にハイパラ・データ拡張・復号設定を
サーバー内で自動探索できる (1 ジョブ 20 時間、再投入で継続)。

```bash
qsub -v "WAV_DIR=data/test_wav,REFS=data/test_refs.txt" scripts/run_autoresearch.pbs
```

## メモ

- LoRA対象は attention の `q/k/v/out_proj`。DDPで unused-param エラーが出る場合は
  `train_whisper_lora.py` の `ddp_find_unused_parameters` を `True` に。
- full-FT で encoder を凍結しても SpecAugment / dropout は掛かったまま(encoder は
  train モードのまま)。decoder 側から見れば入力の揺らぎとして正則化に効く。
- full-FT は optimizer 状態の分だけ VRAM を食う。OOM なら `BATCH_SIZE` を下げて
  `GRAD_ACCUM` を上げる(実効バッチは同じ)。
- 自動探索ループ (`README_autoresearch.md`) は現状 LoRA 空間のみを探索する。
- CT2変換は `merged_hf/` に `tokenizer.json`/`preprocessor_config.json` が保存されている前提
  (train スクリプトが processor を保存する)。
