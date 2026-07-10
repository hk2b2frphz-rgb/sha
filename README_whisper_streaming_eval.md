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
- `experiments/whisper_streaming_eval/summary.json`
- `experiments/whisper_streaming_eval/report.md`

指標:

- CER: 日本語正規化後の文字誤り率
- WER: `fugashi` + `unidic-lite` の分かち書きによる単語誤り率
- speed: 音声長 / 推論wall time
- RTF: 推論wall time / 音声長
