# 評価用データセット (実録音)

自動改善ループ (`README_autoresearch.md`) が読む、実録音のテストデータ置き場。

```
experiments/dataset/
├── wav/          テスト音声 (1.wav, 2.wav, ... 自然順でソートされる)
└── refs.txt      参照テキスト
```

## refs.txt の書き方

2 形式のどちらでもよく、タブの有無で自動判定される (`scripts/build_test_manifest.py`)。

**位置対応 (推奨)** — 1 行 1 正解テキストのみ。wav の自然順 (`1.wav, 2.wav, 10.wav`)
と行番号で対応付ける。

```
心筋梗塞の疑いで搬送された。
下水道管の内部を点検する。
```

**キー付きTSV** — `key<TAB>text`。`key` は wav のファイル名か拡張子なしの名前。

```
1.wav	心筋梗塞の疑いで搬送された。
2.wav	下水道管の内部を点検する。
```

参照テキストがない wav は文字起こしだけ行われ、CER/WER の計算からは外れる
(指標が意味を持たないため)。

## 使われ方

この wav 群は **dev / holdout に自動分割**される (既定 50:50、id のハッシュで決定的)。
探索が最適化に使うのは dev のみ、holdout は最良設定に対して最後に 1 度だけ復号する。
dev だけスコアが良い場合は探索が dev に過適合したと判断できる。

分割の実体は `experiments/autoresearch/data/{dev,holdout}.jsonl` に保存され、
ジョブを再投入しても同じ分割が再利用される。

安定した比較のため **dev 側で最低 30 発話程度** (= 全体で 60 発話程度) を推奨。

## 投入

```bash
qsub scripts/run_autoresearch.pbs
```

`WAV_DIR` / `REFS` の既定値がこのディレクトリを指しているので、上記の配置なら
追加の引数はいらない。別の場所を使う場合:

```bash
qsub -v "WAV_DIR=data/test_wav,REFS=data/test_refs.txt" scripts/run_autoresearch.pbs
```

## 注意: wav は Git に入らない

リポジトリの `.gitignore` に `*.wav` があるため、この `wav/` の中身は
**commit されない**。サーバーへは rsync/scp で直接送るか、Git で運ぶなら
`.gitignore` に例外行を足す:

```
!experiments/dataset/wav/*.wav
```
