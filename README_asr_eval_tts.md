# Evaluation audio generation

`scripts/run_asr_eval_data_tts.pbs` rebuilds only the test WAVs used for
Whisper evaluation. It does not train or alter a Whisper model.

The input sentence file remains in its original Japanese form. Before synthesis,
the PBS passes `data/annotations.tsv` to the preparation step, which replaces
only matching technical terms with their readings. It does not convert the
entire sentence to hiragana. The mapping file can be tab-separated TSV or
comma-separated CSV (including a `.tsv` file containing commas). Supply
another mapping with `READINGS_TSV=...`.

## Neutral male Japanese TTS

To generate the existing test set with Kokoro's Japanese male voice
`jm_kumo`, submit:

```bash
qsub -v "TTS_BACKEND=kokoro,KOKORO_VOICE=jm_kumo" scripts/run_asr_eval_data_tts.pbs
```

The generated files replace the WAVs under
`out/whisper_streaming_eval/audio/wav/`; the source sentences remain in
`data/asr_eval_sentences.txt`. `CLEAN_WAV_DIR=1` is the default, which prevents
old WAVs from accidentally being included in evaluation.

The generated WAVs include 500 ms of leading silence by default. This prevents
the first phoneme from being clipped during playback or streaming evaluation.
Override it with `LEAD_SILENCE_MS`, for example `LEAD_SILENCE_MS=0` to disable
the padding.

`PREPEND_PAUSE=1` is also enabled by default. It prepends a Japanese comma only
to the text sent into the TTS model, causing the model to generate its own
pre-roll before the first phoneme. The comma is not saved in the reference text
or output manifest. Set `PREPEND_PAUSE=0` to disable it.

The default Kokoro voice is the neutral Japanese female voice `jf_alpha`:

```bash
qsub -v "TTS_BACKEND=kokoro,KOKORO_VOICE=jf_alpha" scripts/run_asr_eval_data_tts.pbs
```

To use a different sentence file or output directory, pass `SENTENCES_TXT` or
`OUT_DIR` through `qsub -v` as usual.
