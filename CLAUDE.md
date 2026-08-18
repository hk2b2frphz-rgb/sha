# term2speech / Whisper ドメイン適応

専門用語の ASR 精度を上げるため、TTS で学習データを作り whisper を fine-tune する。

## 最初に読むこと: コードは main に無い

**whisper FT パイプライン一式は `feature/whisper-turbo-ft-tts-20260710` にある。**
`main` は OCR と用語抽出までしか無い。`main` だけ見て「実装が無い」と判断しないこと
(過去に実際にそう誤判断し、既存より劣る重複実装を書いた)。

探すときは全ブランチを見る:

```bash
git fetch origin '+refs/heads/*:refs/remotes/origin/*'
for b in $(git for-each-ref --format='%(refname:short)' refs/remotes/origin/); do
  git grep -il "<探したい語>" "$b" | head -3
done
```

関連リポジトリ `hk2b2frphz-rgb/mil` にも手順が書かれていることがある
(Vast.ai 連携の元ネタは mil の `.claude/hooks/session-start.sh` だった)。

## 実行場所の分担

**GPU を借りるのは学習だけ。** それ以外は CPU コンテナで完結する。

| 工程 | 場所 | スクリプト |
|---|---|---|
| CSV → TTS入力 | CPU | `prepare_train_from_csv.py` |
| TTS 合成 (Kokoro-82M) | CPU | `synthesize_speech_kokoro.py --device cpu` |
| manifest 構築・分割 | CPU | `build_whisper_manifest.py`, `split_whisper_manifest.py` |
| **学習** | **GPU (Vast.ai)** | `run_whisper_train_vast.sh` |
| 推論・評価 | CPU | `eval_whisper_hf.py --device cpu` |

Kokoro は 82M と小さく、4 vCPU で実時間より速い (約 1.5 秒/文)。TTS に GPU は要らない。

```bash
bash scripts/setup_cpu_env.sh                                  # 初回のみ (.venv-cpu)
DATA_REPO=<repo> HF_TOKEN=... bash scripts/run_cpu_dataprep.sh <csv>
HF_TOKEN=... bash scripts/run_whisper_train_vast.sh <data-repo> <out-repo>
```

## 学習の既定は full-FT + encoder 凍結 (LoRA ではない)

学習データが TTS 合成音声なので、encoder まで動かすと合成音の音響特性に過適合して
実音声で崩れる。凍結して decoder だけ更新するのが狙いに合う (`README_whisper_turbo_ft.md`
に論文引用付きで根拠あり)。`FT_MODE=lora` で切替可。

学習率は full=`1e-5` / lora=`1e-4`。LoRA の値を full にそのまま使うと壊れる。

## すべて使い捨て — 残すものは HF へ

**CPU コンテナも借りた GPU も終了時に消える。** ローカルディスクに置いたものは失われる。

| 対象 | 永続する置き場 |
|---|---|
| TTS 音声・manifest | `Tsuka25/term2speech-data` (`DATA_REPO` 指定で自動退避) |
| 学習中の checkpoint | HF `<out-repo>-ckpt` (Vast では既定で有効) |
| 最終モデル (CT2) | HF model repo |

HF ユーザーは **`Tsuka25`**。upload/download とも検証済み (private repo で往復確認)。

`HF_TOKEN` は environment settings 済み。**明示的な `hf auth login` は不要** —
huggingface_hub が環境変数を直接読むので、読み書きとも通る (検証済み)。
session-start hook は `hf` コマンドを入れるだけでよい。

**HF repo は既定で private。** 学習用語は社内用語や未公開の固有名詞であることが多く、
公開すると用語リストごと読める。学習済みモデルからも用語は復元できる。
公開したい場合のみ `PRIVATE=0` を明示する。

manifest の `audio` は生成元マシンの絶対パス。別マシンで学習する際は
`rebase_manifest_paths.py` が貼り替える (ランナーが自動実行)。

## Vast.ai

`VAST_API_KEY` は environment settings 済み。session-start hook が CLI を入れて設定する。

- **借りたら必ず返す**: ランナーは全終了経路で destroy する。放置が一番高い
- `vastai stop` はディスク課金が残る。`destroy` を使う
- GPU は bf16 対応の RTX 3090/4090 に限定 (T4/V100 は bf16 非対応)
- 実測 RTX 3090 約 **$0.10/hr**。`MAX_HOURS=4` で最悪 $0.45
- 借りる前に必ず `DRY_RUN=1` で価格確認し、`LIMIT=20` で小規模通しをする

## codex

**サブスク (ChatGPT 有料プラン) で動く。API キーは不要。**

```bash
npm install -g @openai/codex
codex login --device-auth        # コードを表示 → ブラウザで承認 (ヘッドレス可)
codex exec --skip-git-repo-check -m gpt-5.6-sol "..."
```

**モデルは `gpt-5.6-sol` を指定する。** フラグシップ (最も賢い) モデル。
`-m` を省略すると既定モデルになり sol より弱いので、重要な作業では必ず指定する。

`codex login` のブラウザ認証は使えないが `--device-auth` は使える。
認証情報は `~/.codex/auth.json` に入り、**セッションが切れると消える**。

毎回の認証を省くには environment settings に登録する。session-start hook が復元する。

| 変数 | 中身 | 寿命 |
|---|---|---|
| `CODEX_AUTH_JSON` (推奨) | `base64 -w0 ~/.codex/auth.json` | refresh token を含むので自動更新される |
| `CODEX_ACCESS_TOKEN` | `tokens.access_token` のみ | **約10日で失効** |

```bash
# 手元の codex でログイン済みのマシンで実行し、出力を environment settings へ
base64 -w0 ~/.codex/auth.json        # macOS は base64 -i ~/.codex/auth.json
```

## Claude の使用量

- セッション単位のコスト・トークン: **取得できる** (`claude -p ... --output-format json`
  の `total_cost_usd` / `usage` / `modelUsage`)
- **アカウントの残枠 (5時間/週) は取得できない**。CLI に `usage` サブコマンドは無く、
  `/usage` は対話専用。ローカルにキャッシュも無い。残枠を見るには手元の Claude Code で
  `/usage` を使う

## 評価の読み方

ベースモデルは読みを正しく聞き取るが漢字を間違える (僧帽弁→相棒弁、冠動脈→感動脈)。
`--terms` を渡すと `term_recall` が出る。これが主要指標。

**合成音声だけで学習すると実音声で伸びない可能性がある。評価は実録音で行うこと。**
