# Reading Estimation Method Comparison

This experiment compares reading-estimation methods against human-curated gold
TSV files only. Do not generate gold readings from Sudachi, UniDic,
pyopenjtalk, MeCab, or any other dictionary backend.

## Optional Dictionary Dependencies

The runner attempts these installs automatically and records success/failure in
`experiments/comparison_results.json`:

```bash
python -m pip install pyopenjtalk
python -m pip install sudachipy SudachiDict-core
python -m pip install fugashi unidic-lite
```

If an install fails or a backend cannot be imported, that resolver is skipped
with a clear status line in the JSON and Markdown report.

## Run The Comparison

```bash
python -m experiments.run_method_comparison \
  --out-json experiments/comparison_results.json \
  --out-md experiments/comparison_report.md
```

Use this when optional dependencies were already handled externally:

```bash
python -m experiments.run_method_comparison --no-install
```

The script evaluates all available resolvers over:

- `data/benchmark_terms.tsv` as `domain-A`
- `data/benchmark_terms_sewage_60.tsv` as `domain-B`
- `data/test_cases.tsv` as `general`

Saved LLM resolvers read `experiments/*_neweval.json` and use `llm_output`
only. No live LLM inference is performed.

## Outputs

- `experiments/comparison_results.json`: full per-term results, aggregate
  metrics, bootstrap accuracy intervals, McNemar tests, error analysis, and
  complementarity scores.
- `experiments/comparison_report.md`: paper-oriented Markdown summary with
  design notes, main method-by-domain table, significance tests, error
  breakdowns, complementarity, and key observations.

## Tests

```bash
python -m pytest tests
```

This repository requires Python 3.11+ per `pyproject.toml`. In the current
workspace, the visible system `python` is 3.6; use a Python 3.11+ environment
or another compatible interpreter on PATH.

## Domain-B Benchmark Expansion To 60 Items

Use `data/benchmark_terms_sewage_60_template.tsv` as the curation worksheet.
The template intentionally leaves `term` and `reading` blank. Fill readings only
from independent human review or authoritative domain sources. Do not copy
readings from Sudachi, UniDic/fugashi, pyopenjtalk, saved LLM outputs, or any
future resolver being evaluated; that would make the gold set circular.

Recommended curation workflow:

1. Add domain-B terms to the `term` column.
2. Enter one or more acceptable hiragana readings in `reading`, separated by
   `|` when multiple readings are truly valid.
3. Record the human reviewer or authority in `source`.
4. Use `notes` for abbreviation policy, disputed readings, and source details.
5. After review, copy the curated rows into a benchmark TSV used by the runner,
   preserving the `term` and `reading` columns.

## GPU Server Re-Evaluation With All Methods

On a Linux GPU server, create a Python 3.11+ environment and install optional
dictionary backends including pyopenjtalk:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
python -m pip install pyopenjtalk sudachipy SudachiDict-core fugashi unidic-lite scipy pytest
```

Regenerate or place saved LLM prediction JSON files under `experiments/` using
the same `*_neweval.json` shape expected by `resolvers/llm.py`. Then run:

```bash
python experiments/run_method_comparison.py \
  --out-json experiments/comparison_results.json \
  --out-md experiments/comparison_report.md
```

Confirm `dictionary_status` in `experiments/comparison_results.json` shows
`pyopenjtalk`, `sudachipy`, and `fugashi_unidic` as available. Re-run tests:

```bash
python -m pytest tests
```

## 分野B 60語ベンチをGPUで再評価する手順

Linux GPU server上で、リポジトリのルートから次を実行します。Gemmaの推論は
`gemma_runtime` の `uv` 環境を使い、比較スクリプトは現在のPython環境に
`pyopenjtalk` を入れて実行します。

```bash
python -m pip install -U pip
python -m pip install pyopenjtalk
bash scripts/run_gpu_sewage60.sh
```

スクリプトの中で実行されるコマンド列は次の通りです。

```bash
uv run --project gemma_runtime python scripts/run_eval.py \
  --cases data/benchmark_terms_sewage_60_cases.tsv \
  --provider gemma \
  --model google/gemma-4-E4B-it \
  --out experiments/$(date +%F)_sewage60_e4b_neweval.json

uv run --project gemma_runtime python scripts/run_eval.py \
  --cases data/benchmark_terms_sewage_60_cases.tsv \
  --provider gemma \
  --model google/gemma-4-E2B-it \
  --out experiments/$(date +%F)_sewage60_e2b_neweval.json

python experiments/run_method_comparison.py \
  --out-json experiments/comparison_results.json \
  --out-md experiments/comparison_report.md
```

期待される出力ファイル:

- `experiments/YYYY-MM-DD_sewage60_e4b_neweval.json`
- `experiments/YYYY-MM-DD_sewage60_e2b_neweval.json`
- `experiments/comparison_results.json`
- `experiments/comparison_report.md`

`run_eval.py` の `all_results[].term` と `all_results[].llm_output` は
`resolvers/llm.py` が読む `*_neweval.json` 形式と互換です。同じe4b/e2bの
newevalファイルが複数ある場合は、同一モデルのsaved resolverとしてマージされます。
