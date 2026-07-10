# whisper-large-v3-turbo FT パイプライン (generated_sentences.csv 起点)

`generated_sentences.csv` の専門用語だけをひらがな読みにした発話を Qwen3-TTS で合成し、
その音声を使って whisper-large-v3-turbo を LoRA fine-tune、CT2 へ変換して
whisper-streaming で評価する、ワンパスの手順。

- **学習ターゲット**: 元の `Sentance`(漢字のまま)。TTSはひらがな読みで発話し、whisperには
  「正しい読みの音声 → 漢字表記」を学習させる。
- **FT方式**: LoRA/PEFT (V100 16GB に安全に載る)。学習後 base にマージ→CT2変換。
- **eval対象**: 共有された test wav 群のみ。参照テキストがあれば CER/WER も算出。
- **リソース**: 学習=`res=middle2`(V100×4)、評価=`res=small`(V100×1)。

## 0. 事前準備

- `data/generated_sentences.csv` を配置。列: `Number,Word,Reading,Sentance,AI_Reading,Confirmed_Reading`
- `manifest.txt` の先頭非コメント行を、HPC上の whisper-large-v3-turbo 重みパスに書き換える
  (既定は `openai/whisper-large-v3-turbo`)。

## ネットワーク遮断ノード (res=middle2) 向け: オフライン実行

4GPUノード(`res=middle2`)は**外部ネット非接続**のことが多い。その場合、ネットに出られる
ノード(ログイン or `res=small`)で先にキャッシュを用意してから、学習を `OFFLINE=1` で回す。
共有ファイルシステム(リポジトリ配下 + `~/.cache/huggingface`)経由でmiddle2が読む。

```bash
# 1) ネット可能ノード(ログイン等)で事前DL: .venv + HFキャッシュ(Qwen3-TTS/whisper) + vendor
PROXY_URL=http://user:pass%40@host:port bash scripts/prestage_offline.sh

# 2) 学習はGPUノードでオフライン実行
qsub -v OFFLINE=1 scripts/run_whisper_train.pbs
```

**別案: `res=middle`(2GPU, ネット接続可) でオンライン実行** — prestage不要。GPU枚数は
`NUM_SHARDS` から自動導出するので `CUDA_VISIBLE_DEVICES` を渡す必要はない。

```bash
qsub -l select=1:res=middle -v "NUM_SHARDS=2,PROXY_URL=http://user:pass%40@host:port" scripts/run_whisper_train.pbs
```

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
