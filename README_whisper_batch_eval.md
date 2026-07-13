# Whisper models: batch evaluation

`scripts/run_whisper_eval_all.pbs` evaluates many CTranslate2 (CT2) Whisper
models against exactly the same audio and reference texts. It runs the models
sequentially on one GPU, so results are directly comparable and GPU memory is
not shared between models. All server-specific settings live in the root
`manifest.txt`, which is intentionally ignored by Git.

## 1. Edit `manifest.txt`

Set the test WAV directory, its keyed reference TSV, and one or more models.
Each `MODEL=` value is a tab-separated pair of a unique label and CT2 model
directory. Labels may only contain letters, digits, `.`, `_`, and `-`.

```ini
# Also used by run_whisper_train.pbs.
BASE_MODEL=openai/whisper-large-v3-turbo

WAV_DIR=data/test_wav
REFS=data/test_refs.txt
MODEL=baseline	/path/to/baseline-ct2
MODEL=ft_v1	out/whisper_turbo/ct2
MODEL=ft_v2	out/whisper_turbo_v2/ct2
```

Each listed directory must contain `model.bin`.

## 2. Submit the batch evaluation

Pass the test WAV directory and the matching reference file. `REFS` is
required, ensuring that the comparison reports CER and WER against the ground
truth rather than performing transcription only.

```bash
qsub scripts/run_whisper_eval_all.pbs
```

The reference file should use keyed TSV to avoid accidental WAV-order
mismatches:

```tsv
001.wav	正解文です
002.wav	次の正解文です
```

## Results

Each model's predictions, detailed report, and JSON summary are written to:

```text
experiments/whisper_turbo_eval_all/<label>/
```

The batch also produces these files:

- `experiments/whisper_turbo_eval_all/summary.md` — readable ranking table
- `experiments/whisper_turbo_eval_all/summary.csv` — spreadsheet-friendly data

Models are ranked by lower CER, then lower WER, then higher inference speed.
