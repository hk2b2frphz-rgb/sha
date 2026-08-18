---
name: whisper-ft
description: Run the Whisper domain fine-tuning pipeline — build TTS training data on CPU, rent a Vast.ai GPU for the training only, and evaluate on CPU. Use when asked to fine-tune Whisper, adapt ASR to domain terms, build TTS training data, rent a GPU for training, or evaluate term recall in this repo.
---

# Whisper ドメイン適応の実行手順

専門用語の表記精度を上げるため、TTS で作った音声で whisper-large-v3-turbo を
fine-tune する。**GPU を借りるのは学習だけ**で、他は CPU コンテナで完結する。

## 前提の確認

コード一式は `feature/whisper-turbo-ft-tts-20260710` 系のブランチにある。
`main` には無いので、まず作業ブランチが正しいか確かめる。

```bash
git branch --show-current
ls scripts/train_whisper_lora.py   # 無ければブランチが違う
```

必要なもの:
- `generated_sentences.csv` (列: `Number,Word,Reading,Sentance,...`)
- `HF_TOKEN` (write 権限) — データと成果物の永続化に使う
- `VAST_API_KEY` — session-start hook で設定済みのはず

## 手順

### 1. CPU 環境 (初回のみ)

```bash
bash scripts/setup_cpu_env.sh     # .venv-cpu を作る。pyproject の cu121 固定は触らない
```

### 2. 小規模で通す (必ず先にやる)

いきなり全件回さない。20 件で通してから本番に進む。

```bash
LIMIT=20 bash scripts/run_cpu_dataprep.sh data/generated_sentences.csv
```

### 3. データ生成 (CPU, GPU 不要)

```bash
DATA_REPO=<user>/<data-repo> HF_TOKEN=hf_xxx \
  bash scripts/run_cpu_dataprep.sh data/generated_sentences.csv
```

`DATA_REPO` を必ず渡す。**渡さないとセッション終了で合成結果が消える。**
Kokoro は約 1.5 秒/文なので、1000 文で 25 分ほど。

### 4. ベースラインを測る (CPU)

fine-tune 前の `term_recall` を記録しておかないと、効果を判断できない。

```bash
source .venv-cpu/bin/activate
cut -f2 -d, data/generated_sentences.csv | tail -n +2 > out/terms.txt
python scripts/eval_whisper_hf.py \
    --manifest out/whisper_turbo/dev.jsonl --out-dir out/eval_base \
    --base-model openai/whisper-large-v3-turbo \
    --device cpu --dtype float32 --terms out/terms.txt
```

### 5. 価格を確認してから借りる

```bash
DRY_RUN=1 bash scripts/run_whisper_train_vast.sh <data-repo> <out-repo>
```

提示価格と最悪支出を必ずユーザーに見せ、**借りる前に確認を取る**。

### 6. 学習 (GPU を借りる)

```bash
HF_TOKEN=hf_xxx MAX_HOURS=4 \
  bash scripts/run_whisper_train_vast.sh <data-repo> <out-repo>
```

- 既定は full-FT + encoder 凍結。`FT_MODE=lora` で切替
- checkpoint は `<out-repo>-ckpt` へ自動退避され、中断しても再開できる
- インスタンスは全終了経路で destroy される

### 7. 評価 (CPU)

```bash
hf download <out-repo> --local-dir out/ct2
python scripts/eval_whisper_hf.py \
    --manifest out/whisper_turbo/dev.jsonl --out-dir out/eval_ft \
    --model-dir out/ct2 --device cpu --dtype float32 --terms out/terms.txt
```

手順 4 の `term_recall` と並べて報告する。

## 注意

- **合成音声だけで学習すると実音声で伸びない可能性がある。**
  最終判断は実録音のホールドアウトで行うようユーザーに伝える
- 残高は Vast.ai のクレジット。`vastai show user` で確認できる
- `vastai stop` はディスク課金が残る。終わったら `destroy`
- テストは `python -m pytest tests/ -q` (.venv-cpu で動く)
