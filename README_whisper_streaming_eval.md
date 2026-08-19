# Whisper-Streaming Evaluation

CTranslate2形式のfine-tuned Whisperを、Qwen3-TTSで作った評価音声に対して検証する最小手順です。

## 1. 設定

`configs/whisper_streaming_eval.yaml` の `model.model_dir` にCTranslate2モデルディレクトリを指定します。

PBS投入時に上書きする場合:

```bash
MODEL_DIR=/path/to/ct2-whisper qsub -V scripts/run_whisper_streaming_eval.pbs
```

## 2. 評価音声作成

```bash
qsub -V scripts/run_asr_eval_data_tts.pbs
```

入力例文はデフォルトで `data/asr_eval_sentences.txt` です。差し替え:

```bash
SENTENCES_TXT=/path/to/examples.txt qsub -V scripts/run_asr_eval_data_tts.pbs
```

専門用語の読みを指定する場合は、`term reading` の2列TSVを渡します。
`sentence` は評価用に元の文を保持し、TTSへ渡す `tts_text` だけ `term` を `reading` に置換します。

```bash
SENTENCES_TXT=/path/to/examples.txt READINGS_TSV=/path/to/annotations.tsv qsub -V scripts/run_asr_eval_data_tts.pbs
```

TSV例:

```tsv
term	reading
膜分離活性汚泥法	まくぶんりかっせいおでいほう
嫌気性消化	けんきせいしょうか
```

出力:

- `out/whisper_streaming_eval/audio/wav/*.wav`
- `out/whisper_streaming_eval/audio/manifest.jsonl`

## 3. Whisper-Streaming推論と評価

```bash
MODEL_DIR=/path/to/ct2-whisper qsub -V scripts/run_whisper_streaming_eval.pbs
```

出力:

- `experiments/whisper_streaming_eval/predictions.jsonl`
- `experiments/whisper_streaming_eval/emissions.jsonl`
- `experiments/whisper_streaming_eval/summary.json`
- `experiments/whisper_streaming_eval/report.md`

## emissions.jsonl（リアルタイム性の可視化用）

逐次デコード中に確定した文字列を、確定した時点の情報つきで1発話1行で記録します。
`predictions.jsonl` を読みやすい大きさに保つため別ファイルにし、1確定あたりの情報は
キー名を持たない配列にしています。

```json
{"id": "0001", "duration_sec": 4.2, "emits": [[1.31, 0.0, 1.2, "活性汚泥法の"], [2.68, 1.2, 2.5, "運転管理では"]]}
```

`emits` の各要素は `[経過実時間(秒), 区間開始(秒), 区間終了(秒), 確定テキスト]` です。

- 経過実時間: その発話の処理を開始してから確定するまでの秒数
- 区間開始 / 区間終了: 確定テキストが対応する音声上の位置（`process_iter()` の戻り値）
- 遅延: `経過実時間 - 区間終了`
- 最後の要素は `finish()` によるflush分なので、末尾の遅延はここに現れます

`predictions.jsonl` 側にも要約値として `emission_count`、`first_emission_sec`、
`final_emission_lag_sec` を持たせています。

指標:

- CER: 日本語正規化後の文字誤り率
- WER: `fugashi` + `unidic-lite` の分かち書きによる単語誤り率
- insertions / insertion rate: 参照にない文字の挿入数・率
- empty-reference false positives: 無音・雑音で出力された発話数と文字数
- repeated n-grams: 仮説内の反復数・率
- speed: 音声長 / 推論wall time
- RTF: 推論wall time / 音声長

既定設定は湧き出し対策としてVADを有効にし、前chunkの内部履歴と確定文字列のprompt再注入を
どちらも無効にして、temperatureを0に固定します。さらにno-speech/log-prob/compression-ratio閾値、反復抑制、生成token上限を
`configs/whisper_streaming_eval.yaml` から調整できます。インストール済みfaster-whisperが
古く未対応の項目は警告付きで除外され、`summary.json` に項目名が記録されます。
