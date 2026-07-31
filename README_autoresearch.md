# Whisper 自動改善ループ (auto-research)

サーバー上の 1 ジョブの中で、**「設定を提案 → LoRA 学習 → dev セットで復号 → CER 採点 → 履歴に記録」**
を制限時間まで自動で反復する探索ループ。手元からサーバーに触れなくても、`git pull` → `qsub`
だけで改善サイクルが回る。

- 最適化対象: **whisper-large-v3-turbo の日本語 ASR 精度 (CER)**
- 探索: **2 段構え**。短時間の代理試行で多数を篩い、上位のみフル学習で再評価
- 提案器: **ローカル Gemma** (`gemma_runtime` env)。**フォールバックなし**（失敗したら止まる）
- 時間管理: walltime 24h に対し **既定 20h** でループを打ち切り、残りで最終評価と書き出し
- 再開: 同じ `WORK_DIR` に `qsub` し直すと、終わった試行を引き継いで続きから探索する

参考にした OSS: [AIDE (WecoAI)](https://github.com/WecoAI/aideml) — 過去の試行と
スコアを LLM に見せて次の候補を書かせる木探索型 ML エージェント。
[OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) — 評価器を軸にエリート集団を
変異・交叉させる AlphaEvolve 系の進化的コード改善。本ループは 1 試行が「LoRA 学習 +
復号」で重い（数十分〜数時間）ため、**数十試行しか回せない前提**で両者を折衷している:
序盤はランダム探索で当たりを付け、以降は Gemma に提案させる。
[karpathy/autoresearch](https://github.com/karpathy/autoresearch) — 1 試行を 5 分固定にして
一晩で約 100 回回す構成。ここからは「試行あたりの時間を固定して試行数を稼ぐ」考え方と、
人間の申し送りファイル (`program.md` 相当 = `PROPOSER_GUIDANCE.md`) を取り入れた。

## 1. 前提データ

| 必要なもの | 既定パス | 用意の仕方 |
|---|---|---|
| 学習データ (TTS 音声 + 漢字正解) | `out/whisper_turbo/train_manifest.jsonl` | `qsub scripts/run_whisper_train.pbs` を 1 度流す (このファイルがあれば TTS は再利用される) |
| テストデータ (実録音 wav) | `experiments/dataset/wav/` | 録音ツール `tools/record_test_wav.html` などで用意 |
| 参照テキスト | `experiments/dataset/refs.txt` | 1 行 1 正解テキスト (wav の自然順) か `key<TAB>text` |

配置の詳細は [experiments/dataset/README.md](experiments/dataset/README.md)。
`.gitignore` の `*.wav` により wav は commit されないので、サーバーへは rsync/scp で送るか
`.gitignore` に例外を足す。

テストデータは **dev / holdout に自動分割**される (既定 50:50、id のハッシュで決定的)。
ループが最適化に使うのは dev のみで、holdout は最後に 1 度だけ復号する。
探索が dev に過適合したかどうかを、その holdout スコアで検出できるようにしてある。

## 2. 投入

`experiments/dataset/` に置いた場合は引数なしで投げられる。

```bash
qsub scripts/run_autoresearch.pbs
```

```bash
qsub -v "BUDGET_HOURS=20,NUM_GPUS=4,OBJECTIVE=cer" scripts/run_autoresearch.pbs
```

```bash
qsub -v "WAV_DIR=data/test_wav,REFS=data/test_refs.txt" scripts/run_autoresearch.pbs
```

```bash
qsub -q xan_s -l select=1:res=middle -v NUM_GPUS=2 scripts/run_autoresearch.pbs
```

ネット遮断ノード (`res=middle2`) では、ネットに出られるノードで
`scripts/prestage_offline.sh` を先に流す (`.venv` / HF キャッシュに加えて、提案器用の
`gemma_runtime` env と Gemma 本体もキャッシュされる。不要なら `PRESTAGE_GEMMA=0`)。
その上で:

```bash
qsub -v OFFLINE=1 scripts/run_autoresearch.pbs
```

主な `qsub -v` 変数:

| 変数 | 既定 | 意味 |
|---|---|---|
| `BUDGET_HOURS` | `20` | ループに使う時間。walltime 24h の内側に収める |
| `RESERVE_MINUTES` | `40` | 最終 holdout 評価 + CT2 変換のために残す時間 |
| `MAX_TRIAL_HOURS` | `6` | 1 試行の上限。超えた試行は打ち切って `timeout` 記録 |
| `NUM_GPUS` | `4` | 学習の DDP プロセス数 (= 使う GPU 枚数) |
| `OBJECTIVE` | `cer` | `cer` / `wer` / `cer_term` (専門用語の再現率を加味) |
| `USE_GEMMA` | `1` | `0` で LLM 提案を止め、進化的探索のみにする |
| `GEMMA_MODEL` | `google/gemma-4-E4B-it` | 提案器のモデル |
| `DEV_RATIO` | `0.5` | テストデータのうち dev に回す割合 |
| `MAX_TRIALS` | `0` | `0` は時間の許す限り |
| `MAX_CONSECUTIVE_FAILURES` | `5` | 連続失敗でループを中断 (`0` で無効) |
| `SCREEN_MINUTES` | `25` | 代理試行 1 件あたりの学習時間。`0` で 2 段構成を無効化 |
| `SCREEN_BUDGET_FRACTION` | `0.6` | 代理試行に充てる予算の割合 |
| `FINALISTS` | `3` | フル学習で再評価する上位件数 |
| `GUIDANCE` | `PROPOSER_GUIDANCE.md` | 提案器への申し送りファイル |
| `EARLY_STOP_MARGIN` | `0.35` | 学習中の足切り水準 = ベスト×(1+margin)。`0` で無効 |
| `EARLY_STOP_PATIENCE` | `2` | dev CER が改善しないエポックがこの回数続いたら打ち切り |
| `EARLY_STOP_DEV_LIMIT` | `24` | 学習中の評価に使う発話数 |
| `WRITE_ANALYSIS` | `1` | 探索後に Gemma が `report.md` に考察を書く。`0` で無効 |
| `FINAL_EXPORT` | `1` | 最良 LoRA を base にマージ + CT2 変換まで行う |
| `TERMS` | (自動) | 専門用語リスト。既定は学習データ側から自動抽出 |

## 3. 探索空間

`autoresearch/space.py` に宣言的に定義。ランダム生成・変異・LLM 提案の検証・
プロンプト文面がすべて同じ定義から作られるので、**空間外の設定は実行されない**
(LLM が範囲外を返しても最寄りの合法値に丸められる)。

- 最適化: `epochs` / `lr` / `batch_size` / `grad_accum` / `warmup_ratio` / `weight_decay` /
  `label_smoothing` / `lr_scheduler`
- LoRA 容量: `lora_r` / `lora_alpha_mult` / `lora_dropout` / `target_modules`
  (`qv` / `qkvo` / `qkvo_fc`)
- データ拡張: `augment_copies` / `augment_speed` / `augment_noise_snr_db` /
  `augment_gain_db` / `spec_augment`
- 復号: `beam_size` / `no_repeat_ngram_size` / `prompt_terms`
  (専門用語を Whisper の initial prompt として与えるか)

試行 0 は必ず**現行の既定値** (`run_whisper_train.pbs` と同じハイパラ) を評価するので、
レポートの改善率は常にベースライン比で読める。

## 4. 1 試行の中身

```
scripts/train_whisper_lora.py   # LoRA 学習 (マージも CT2 変換もしない)
scripts/eval_whisper_hf.py      # base + adapter を直接読んで dev をバッチ復号 → CER/WER/用語再現率
```

重みマージと CT2 変換は最後の 1 回だけ (`scripts/export_whisper_model.py`)。
これで 1 試行が「学習 + 復号」だけになり、20 時間に収まる試行数が増える。
CER/WER の計算は `eval/asr_text.py` に集約してあり、既存の
`evaluate_whisper_streaming.py` と同じ正規化・同じ編集距離を使う
(復号方式はバッチ vs ストリーミングで異なる点に注意)。

## 5. 2 段探索 (代理試行 -> 本番)

探索の精度は試行数で決まる。1 試行をフル学習にすると一晩で 10〜30 回しか回らないので、
**前半は学習時間を固定した代理試行**で数を稼ぎ、**後半で上位だけをフル学習**し直す。

```
stage 1 (screening)  各試行 SCREEN_MINUTES 分だけ学習 -> dev 復号 -> 採点
                     BUDGET_HOURS x SCREEN_BUDGET_FRACTION を使い切るまで反復
stage 2 (finals)     stage 1 の上位 FINALISTS 件を、時間制限なしで学習し直す
```

学習の打ち切りは**実時間**で行う (`--max-train-seconds`)。ステップ数で揃えると
`batch_size` や `augment_copies` が違う設定同士で計算量が揃わないため。

**2 つのステージのスコアは比較できない**（学習量が違う）。そのため:

- 順位表・早期打ち切りのしきい値・エリート選抜は、**同じステージ内でのみ**比較する
- レポートの見出しスコア、`best_config.json`、holdout 評価、CT2 書き出しは、
  **フル学習の試行からのみ**選ばれる
- Gemma に見せる履歴にも `[screen]` / `[full]` のタグを付ける

代理試行の順位がフル学習でひっくり返ることは実際に起きる。stage 2 はそのための保険で、
`FINALISTS` を増やすほど取りこぼしは減るが、その分フル学習の回数が増える。

`SCREEN_MINUTES=0` で 2 段構成を無効化し、従来どおり全試行をフル学習にできる。

## 6. 提案器への申し送り (PROPOSER_GUIDANCE.md)

`PROPOSER_GUIDANCE.md` に書いた内容は、そのまま Gemma の提案プロンプトに差し込まれる。
探索が自力で気づけない前提（OOM しやすい設定、運用上の制約、過去に効かなかった設定）を
書いておく場所。空なら何も渡さない。`GUIDANCE=` で別ファイルを指定できる。

## 7. フォールバックを置かない方針

「動いているのか分からない」状態を作らないため、**失敗を黙って回避する経路を持たない**。

| 起きたこと | 挙動 |
|---|---|
| Gemma の提案が失敗 (異常終了/タイムアウト/JSON 不正/重複のみ) | 2 回まで再試行し、それでも駄目ならエラーを出して**ループ停止** |
| Gemma が範囲外の値を返す | 最寄りの合法値に丸め、**修正内容をログに全部出す** |
| 考察の生成が失敗 | レポートに理由を書き、**ジョブは exit 1** |
| 試行が失敗 (OOM 等) | 実験結果として記録して次へ (連続 `MAX_CONSECUTIVE_FAILURES` 回で停止) |

進化的探索 (`--no-gemma`) は**明示的に選ぶモード**であって、LLM が失敗したときの
代替ではない。LLM 抜きで回したいときだけ使う。

## 8. 見込みのない試行の早期打ち切り

学習中、**毎エポックごとに dev の一部 (既定24発話) を復号して CER を測り**、
見込みがなければその場で学習を止めて次の試行に進む。同じ20時間で試せる試行数を
増やすための仕組み。

打ち切りの条件は 2 つで、どちらか一方でも該当したら止める。

1. **足切り**: その時点の dev CER が `ベストスコア × (1 + EARLY_STOP_MARGIN)` を超えた
   (既定 1.35 倍)。ベストがまだ無い最初の試行では発動しない
2. **頭打ち**: dev CER が `EARLY_STOP_PATIENCE` 回連続で改善しなかった。
   `epochs` を大きく取った設定が、伸びなくなった後も回り続けるのを防ぐ

学習中の評価は dev の一部しか使わない粗い推定なので、マージンを広めに取って
**「明らかに悪い」ものだけを切る**設定にしてある。切りすぎていると感じたら
`EARLY_STOP_MARGIN` を上げるか、`0` で機能ごと止められる。

打ち切られた試行も**破棄はせず**、その時点のアダプタで dev 全体を評価して通常どおり
記録する (score は当然悪くなる)。CER 推移と打ち切り理由は
`trials/trial_XXXX/train_progress.json` と `report.md` に残る。

なお `OBJECTIVE=cer_term` のときスコアは CER 以上の値になるため、しきい値は自然に
甘い側 (打ち切りにくい側) に倒れる。

## 9. 出力

```
experiments/autoresearch/
  report.md                 # 順位表 + 上位試行に多い設定値 + ベースライン比の改善率
  best_config.json          # 最良設定
  state/state.json          # 再開用の全履歴 (毎試行ごとに原子的に書き換え)
  state/trials.jsonl        # 追記ログ
  trials/trial_XXXX/        # config.json, trial.log, adapter/(上位のみ保持), eval/
  proposals/                # Gemma に投げた要求と生の応答
  final/holdout_summary.json# 最良設定を holdout で 1 度だけ評価した結果
  final/analysis_*.json     # 考察を書かせたときの要求と応答
  final/ct2/                # FINAL_EXPORT=1 のとき。既存の評価 PBS にそのまま渡せる
```

`report.md` の内容:

1. **順位表** — 試行ごとの score / CER / WER / 用語再現率 / 所要分数 (機械集計)
2. **上位試行に多い設定値** — 各ノブが上位で何回選ばれたかの分布 (機械集計)
3. **最良設定** の JSON
4. **考察** — 探索終了後に Gemma が 1〜3 の表だけを読んで書く日本語の文章。
   「効いた設定 / 判断できない設定 / 次に試すべきこと / 注意点」の 4 節。
   `WRITE_ANALYSIS=0` で省略できる

考察は**検証されていない文章**なので、`report.md` 上でもその旨の注記付きで
機械集計と明確に分けてある。数値の根拠は必ず順位表と `state/trials.jsonl` で確認すること。
生成に失敗しても、レポートから考察が落ちるだけで他は変わらない。

ディスク対策として、adapter は上位 `KEEP_ADAPTERS` 件 (既定 3) だけ残し、
それ以外は学習後に削除する (ログと設定は全件残る)。

最終モデルを既存のストリーミング評価に掛ける場合:

```bash
qsub -v "MODEL_DIR=experiments/autoresearch/final/ct2,WAV_DIR=data/test_wav,REFS=data/test_refs.txt" scripts/run_whisper_eval.pbs
```

## 10. 途中経過の見方

```bash
tail -f experiments/pbs_logs/ws_autoresearch_*.log
```

```bash
cat experiments/autoresearch/report.md
```

`report.md` は試行が 1 つ終わるたびに書き直されるので、ジョブ実行中でも現在の順位が読める。

## 11. 中断・再開

walltime 切れや `qdel` で SIGTERM が飛ぶと、進行中の試行を終えた時点で状態を保存して
終了する。同じ `WORK_DIR` に再投入すれば履歴を引き継いで続きから探索する:

```bash
qsub scripts/run_autoresearch.pbs
```

dev/holdout の分割も保存済みのものを再利用するので、セッションをまたいでスコアが
比較可能なまま保たれる。

## 12. 手元での動作確認 (GPU 不要)

実際の学習・復号の代わりに合成スコアを返すモードがある。配線・再開・レポート生成の
確認用:

```bash
python scripts/run_autoresearch.py --mock --train-manifest <train.jsonl> --eval-manifest <test.jsonl> --work-dir /tmp/ar --max-trials 8
```

```bash
python -m pytest tests -q
```

## 13. サーバー上での動作確認 (スモークテスト)

合成音声のダミーデータで**実機のパイプライン全体**を一度通す PBS。学習・復号・Gemma 提案・
2 段探索・holdout 評価・考察生成・CT2 書き出しまで、実物を動かして出力の有無を検査する。

```bash
qsub scripts/run_autoresearch_smoke.pbs
```

```bash
qsub -v USE_GEMMA=0 scripts/run_autoresearch_smoke.pbs
```

- 音声は合成波形（人の声ではない）なので、**CER は 1.0 付近になるのが正常**。
  見ているのは「各段が動いてファイルを吐くか」であって精度ではない
- 既存の成果物で誤って通らないよう、開始時に `SMOKE_DIR` を消してから作り直す
- 各段の出力を個別に検査し、1 つでも欠ければ `SMOKE TEST FAILED` で **exit 1**
- 所要は 1〜2 時間程度（大半は whisper と Gemma のロード）

本番投入の前に、まずこれを通すこと。

## 14. 注意点

- Gemma 提案は **GPU を使う**。学習と同時には走らないが、`GEMMA_MODEL` が大きいと
  1 提案あたり数分かかる。提案は 3 試行に 1 回（`GEMMA_PROPOSALS` 件まとめて受け取る）。
- 代理試行の学習時間は `SCREEN_MINUTES` で固定されるが、モデルのロードと dev 復号は
  別途かかる。1 試行の実測は `SCREEN_MINUTES + 数分` になる。
- 探索は dev セット (実録音の半分) に対して行う。テストデータが少ないと
  スコアのばらつきが探索の障害になるので、**dev は最低でも 30 発話程度**を推奨。
- 1 試行の所要時間は学習データ量と GPU 枚数で決まる。ループは直近 5 試行の実測値から
  残り時間を見積もり、入らないと判断したら次の試行を始めない。
