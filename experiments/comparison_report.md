# Japanese Reading Estimation Method Comparison

## Design Notes

- Gold readings are loaded only from the human-curated TSV files in `data/`; dictionary analyzers are prediction-side methods only.
- LLM methods use saved `experiments/*_neweval.json` predictions and do not run live inference.
- First-round hybrid methods use a dictionary prediction only when it is single-token/high-confidence or when dictionary and LLM readings agree; otherwise they fall back to the saved LLM output.
- Smart hybrid methods score prediction-side signals only: OOV/unknown flags, kana consistency, tokenization confidence, multiple dictionary agreement, saved-LLM agreement, and surface lexical markers. They do not read gold labels.
- Optional dictionary dependencies are skipped when unavailable, with install commands recorded below.

## Optional Dictionary Status

| Resolver | Available | Message | Install |
|---|---:|---|---|
| pyopenjtalk | False | skipped: ModuleNotFoundError: No module named 'pyopenjtalk' | `python -m pip install pyopenjtalk` |
| sudachipy | True | available | `python -m pip install sudachipy SudachiDict-core` |
| fugashi_unidic | True | available | `python -m pip install fugashi unidic-lite` |

## Optional Install Attempts

| Backend | Attempted | Success | Message |
|---|---:|---:|---|

## Main Results

| Method | Domain | n | Accuracy | 95% CI | MER | Char Edit |
|---|---|---:|---:|---|---:|---:|
| sudachipy | medical | 15 | 0.867 | [0.667, 1.000] | 0.043 | 0.333 |
| sudachipy | sewage | 60 | 0.833 | [0.733, 0.917] | 0.058 | 0.400 |
| sudachipy | general | 30 | 0.800 | [0.667, 0.900] | 0.061 | 0.467 |
| fugashi_unidic | medical | 15 | 0.800 | [0.600, 0.933] | 0.096 | 1.133 |
| fugashi_unidic | sewage | 60 | 0.833 | [0.733, 0.917] | 0.067 | 0.433 |
| fugashi_unidic | general | 30 | 0.767 | [0.633, 0.867] | 0.088 | 0.867 |
| gemma_e2b_saved | medical | 15 | 0.800 | [0.600, 1.000] | 0.034 | 0.467 |
| gemma_e2b_saved | sewage | 60 | 0.100 | [0.017, 0.167] | 0.788 | 6.150 |
| gemma_e2b_saved | general | 30 | 0.600 | [0.433, 0.800] | 0.092 | 0.967 |
| gemma_e4b_saved | medical | 15 | 1.000 | [1.000, 1.000] | 0.000 | 0.000 |
| gemma_e4b_saved | sewage | 60 | 0.167 | [0.067, 0.250] | 0.766 | 5.917 |
| gemma_e4b_saved | general | 30 | 0.833 | [0.700, 0.933] | 0.031 | 0.267 |
| hybrid_sudachipy+gemma_e2b_saved | medical | 15 | 0.733 | [0.533, 0.933] | 0.067 | 0.667 |
| hybrid_sudachipy+gemma_e2b_saved | sewage | 60 | 0.183 | [0.100, 0.267] | 0.692 | 5.783 |
| hybrid_sudachipy+gemma_e2b_saved | general | 30 | 0.567 | [0.400, 0.733] | 0.109 | 1.067 |
| hybrid_sudachipy+gemma_e4b_saved | medical | 15 | 0.933 | [0.800, 1.000] | 0.033 | 0.200 |
| hybrid_sudachipy+gemma_e4b_saved | sewage | 60 | 0.250 | [0.150, 0.333] | 0.670 | 5.550 |
| hybrid_sudachipy+gemma_e4b_saved | general | 30 | 0.800 | [0.633, 0.933] | 0.048 | 0.367 |
| smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | medical | 15 | 1.000 | [1.000, 1.000] | 0.000 | 0.000 |
| smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | sewage | 60 | 0.883 | [0.817, 0.950] | 0.043 | 0.283 |
| smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | general | 30 | 0.967 | [0.900, 1.000] | 0.008 | 0.067 |

## McNemar Exact-Match Tests

### medical

| Method A | Method B | A fail/B success | A success/B fail | chi2 | p |
|---|---|---:|---:|---:|---:|
| sudachipy | fugashi_unidic | 0 | 1 | 0.000 | 1.000 |
| sudachipy | gemma_e2b_saved | 1 | 2 | 0.000 | 1.000 |
| sudachipy | gemma_e4b_saved | 2 | 0 | 0.500 | 0.479 |
| sudachipy | hybrid_sudachipy+gemma_e2b_saved | 0 | 2 | 0.500 | 0.479 |
| sudachipy | hybrid_sudachipy+gemma_e4b_saved | 1 | 0 | 0.000 | 1.000 |
| sudachipy | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 2 | 0 | 0.500 | 0.479 |
| fugashi_unidic | gemma_e2b_saved | 1 | 1 | 0.500 | 0.479 |
| fugashi_unidic | gemma_e4b_saved | 3 | 0 | 1.333 | 0.248 |
| fugashi_unidic | hybrid_sudachipy+gemma_e2b_saved | 0 | 1 | 0.000 | 1.000 |
| fugashi_unidic | hybrid_sudachipy+gemma_e4b_saved | 2 | 0 | 0.500 | 0.479 |
| fugashi_unidic | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 3 | 0 | 1.333 | 0.248 |
| gemma_e2b_saved | gemma_e4b_saved | 3 | 0 | 1.333 | 0.248 |
| gemma_e2b_saved | hybrid_sudachipy+gemma_e2b_saved | 0 | 1 | 0.000 | 1.000 |
| gemma_e2b_saved | hybrid_sudachipy+gemma_e4b_saved | 3 | 1 | 0.250 | 0.617 |
| gemma_e2b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 3 | 0 | 1.333 | 0.248 |
| gemma_e4b_saved | hybrid_sudachipy+gemma_e2b_saved | 0 | 4 | 2.250 | 0.134 |
| gemma_e4b_saved | hybrid_sudachipy+gemma_e4b_saved | 0 | 1 | 0.000 | 1.000 |
| gemma_e4b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 0 | 0 | 0.000 | 1.000 |
| hybrid_sudachipy+gemma_e2b_saved | hybrid_sudachipy+gemma_e4b_saved | 3 | 0 | 1.333 | 0.248 |
| hybrid_sudachipy+gemma_e2b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 4 | 0 | 2.250 | 0.134 |
| hybrid_sudachipy+gemma_e4b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 1 | 0 | 0.000 | 1.000 |

### sewage

| Method A | Method B | A fail/B success | A success/B fail | chi2 | p |
|---|---|---:|---:|---:|---:|
| sudachipy | fugashi_unidic | 0 | 0 | 0.000 | 1.000 |
| sudachipy | gemma_e2b_saved | 2 | 46 | 38.521 | 0.000 |
| sudachipy | gemma_e4b_saved | 2 | 42 | 34.568 | 0.000 |
| sudachipy | hybrid_sudachipy+gemma_e2b_saved | 2 | 41 | 33.581 | 0.000 |
| sudachipy | hybrid_sudachipy+gemma_e4b_saved | 2 | 37 | 29.641 | 0.000 |
| sudachipy | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 3 | 0 | 1.333 | 0.248 |
| fugashi_unidic | gemma_e2b_saved | 2 | 46 | 38.521 | 0.000 |
| fugashi_unidic | gemma_e4b_saved | 2 | 42 | 34.568 | 0.000 |
| fugashi_unidic | hybrid_sudachipy+gemma_e2b_saved | 2 | 41 | 33.581 | 0.000 |
| fugashi_unidic | hybrid_sudachipy+gemma_e4b_saved | 2 | 37 | 29.641 | 0.000 |
| fugashi_unidic | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 3 | 0 | 1.333 | 0.248 |
| gemma_e2b_saved | gemma_e4b_saved | 4 | 0 | 2.250 | 0.134 |
| gemma_e2b_saved | hybrid_sudachipy+gemma_e2b_saved | 5 | 0 | 3.200 | 0.074 |
| gemma_e2b_saved | hybrid_sudachipy+gemma_e4b_saved | 9 | 0 | 7.111 | 0.008 |
| gemma_e2b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 47 | 0 | 45.021 | 0.000 |
| gemma_e4b_saved | hybrid_sudachipy+gemma_e2b_saved | 5 | 4 | 0.000 | 1.000 |
| gemma_e4b_saved | hybrid_sudachipy+gemma_e4b_saved | 5 | 0 | 3.200 | 0.074 |
| gemma_e4b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 43 | 0 | 41.023 | 0.000 |
| hybrid_sudachipy+gemma_e2b_saved | hybrid_sudachipy+gemma_e4b_saved | 4 | 0 | 2.250 | 0.134 |
| hybrid_sudachipy+gemma_e2b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 42 | 0 | 40.024 | 0.000 |
| hybrid_sudachipy+gemma_e4b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 38 | 0 | 36.026 | 0.000 |

### general

| Method A | Method B | A fail/B success | A success/B fail | chi2 | p |
|---|---|---:|---:|---:|---:|
| sudachipy | fugashi_unidic | 0 | 1 | 0.000 | 1.000 |
| sudachipy | gemma_e2b_saved | 3 | 9 | 2.083 | 0.149 |
| sudachipy | gemma_e4b_saved | 4 | 3 | 0.000 | 1.000 |
| sudachipy | hybrid_sudachipy+gemma_e2b_saved | 2 | 9 | 3.273 | 0.070 |
| sudachipy | hybrid_sudachipy+gemma_e4b_saved | 3 | 3 | 0.167 | 0.683 |
| sudachipy | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 5 | 0 | 3.200 | 0.074 |
| fugashi_unidic | gemma_e2b_saved | 3 | 8 | 1.455 | 0.228 |
| fugashi_unidic | gemma_e4b_saved | 5 | 3 | 0.125 | 0.724 |
| fugashi_unidic | hybrid_sudachipy+gemma_e2b_saved | 2 | 8 | 2.500 | 0.114 |
| fugashi_unidic | hybrid_sudachipy+gemma_e4b_saved | 4 | 3 | 0.000 | 1.000 |
| fugashi_unidic | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 6 | 0 | 4.167 | 0.041 |
| gemma_e2b_saved | gemma_e4b_saved | 7 | 0 | 5.143 | 0.023 |
| gemma_e2b_saved | hybrid_sudachipy+gemma_e2b_saved | 0 | 1 | 0.000 | 1.000 |
| gemma_e2b_saved | hybrid_sudachipy+gemma_e4b_saved | 7 | 1 | 3.125 | 0.077 |
| gemma_e2b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 11 | 0 | 9.091 | 0.003 |
| gemma_e4b_saved | hybrid_sudachipy+gemma_e2b_saved | 0 | 8 | 6.125 | 0.013 |
| gemma_e4b_saved | hybrid_sudachipy+gemma_e4b_saved | 0 | 1 | 0.000 | 1.000 |
| gemma_e4b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 4 | 0 | 2.250 | 0.134 |
| hybrid_sudachipy+gemma_e2b_saved | hybrid_sudachipy+gemma_e4b_saved | 7 | 0 | 5.143 | 0.023 |
| hybrid_sudachipy+gemma_e2b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 12 | 0 | 10.083 | 0.001 |
| hybrid_sudachipy+gemma_e4b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 5 | 0 | 3.200 | 0.074 |

## Exact McNemar Binomial Tests

### medical

| Method A | Method B | A fail/B success | A success/B fail | p | Method |
|---|---|---:|---:|---:|---|
| sudachipy | fugashi_unidic | 0 | 1 | 1.000 | scipy.stats.binomtest |
| sudachipy | gemma_e2b_saved | 1 | 2 | 1.000 | scipy.stats.binomtest |
| sudachipy | gemma_e4b_saved | 2 | 0 | 0.500 | scipy.stats.binomtest |
| sudachipy | hybrid_sudachipy+gemma_e2b_saved | 0 | 2 | 0.500 | scipy.stats.binomtest |
| sudachipy | hybrid_sudachipy+gemma_e4b_saved | 1 | 0 | 1.000 | scipy.stats.binomtest |
| sudachipy | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 2 | 0 | 0.500 | scipy.stats.binomtest |
| fugashi_unidic | gemma_e2b_saved | 1 | 1 | 1.000 | scipy.stats.binomtest |
| fugashi_unidic | gemma_e4b_saved | 3 | 0 | 0.250 | scipy.stats.binomtest |
| fugashi_unidic | hybrid_sudachipy+gemma_e2b_saved | 0 | 1 | 1.000 | scipy.stats.binomtest |
| fugashi_unidic | hybrid_sudachipy+gemma_e4b_saved | 2 | 0 | 0.500 | scipy.stats.binomtest |
| fugashi_unidic | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 3 | 0 | 0.250 | scipy.stats.binomtest |
| gemma_e2b_saved | gemma_e4b_saved | 3 | 0 | 0.250 | scipy.stats.binomtest |
| gemma_e2b_saved | hybrid_sudachipy+gemma_e2b_saved | 0 | 1 | 1.000 | scipy.stats.binomtest |
| gemma_e2b_saved | hybrid_sudachipy+gemma_e4b_saved | 3 | 1 | 0.625 | scipy.stats.binomtest |
| gemma_e2b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 3 | 0 | 0.250 | scipy.stats.binomtest |
| gemma_e4b_saved | hybrid_sudachipy+gemma_e2b_saved | 0 | 4 | 0.125 | scipy.stats.binomtest |
| gemma_e4b_saved | hybrid_sudachipy+gemma_e4b_saved | 0 | 1 | 1.000 | scipy.stats.binomtest |
| gemma_e4b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 0 | 0 | 1.000 | no_discordance |
| hybrid_sudachipy+gemma_e2b_saved | hybrid_sudachipy+gemma_e4b_saved | 3 | 0 | 0.250 | scipy.stats.binomtest |
| hybrid_sudachipy+gemma_e2b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 4 | 0 | 0.125 | scipy.stats.binomtest |
| hybrid_sudachipy+gemma_e4b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 1 | 0 | 1.000 | scipy.stats.binomtest |

### sewage

| Method A | Method B | A fail/B success | A success/B fail | p | Method |
|---|---|---:|---:|---:|---|
| sudachipy | fugashi_unidic | 0 | 0 | 1.000 | no_discordance |
| sudachipy | gemma_e2b_saved | 2 | 46 | 0.000 | scipy.stats.binomtest |
| sudachipy | gemma_e4b_saved | 2 | 42 | 0.000 | scipy.stats.binomtest |
| sudachipy | hybrid_sudachipy+gemma_e2b_saved | 2 | 41 | 0.000 | scipy.stats.binomtest |
| sudachipy | hybrid_sudachipy+gemma_e4b_saved | 2 | 37 | 0.000 | scipy.stats.binomtest |
| sudachipy | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 3 | 0 | 0.250 | scipy.stats.binomtest |
| fugashi_unidic | gemma_e2b_saved | 2 | 46 | 0.000 | scipy.stats.binomtest |
| fugashi_unidic | gemma_e4b_saved | 2 | 42 | 0.000 | scipy.stats.binomtest |
| fugashi_unidic | hybrid_sudachipy+gemma_e2b_saved | 2 | 41 | 0.000 | scipy.stats.binomtest |
| fugashi_unidic | hybrid_sudachipy+gemma_e4b_saved | 2 | 37 | 0.000 | scipy.stats.binomtest |
| fugashi_unidic | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 3 | 0 | 0.250 | scipy.stats.binomtest |
| gemma_e2b_saved | gemma_e4b_saved | 4 | 0 | 0.125 | scipy.stats.binomtest |
| gemma_e2b_saved | hybrid_sudachipy+gemma_e2b_saved | 5 | 0 | 0.062 | scipy.stats.binomtest |
| gemma_e2b_saved | hybrid_sudachipy+gemma_e4b_saved | 9 | 0 | 0.004 | scipy.stats.binomtest |
| gemma_e2b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 47 | 0 | 0.000 | scipy.stats.binomtest |
| gemma_e4b_saved | hybrid_sudachipy+gemma_e2b_saved | 5 | 4 | 1.000 | scipy.stats.binomtest |
| gemma_e4b_saved | hybrid_sudachipy+gemma_e4b_saved | 5 | 0 | 0.062 | scipy.stats.binomtest |
| gemma_e4b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 43 | 0 | 0.000 | scipy.stats.binomtest |
| hybrid_sudachipy+gemma_e2b_saved | hybrid_sudachipy+gemma_e4b_saved | 4 | 0 | 0.125 | scipy.stats.binomtest |
| hybrid_sudachipy+gemma_e2b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 42 | 0 | 0.000 | scipy.stats.binomtest |
| hybrid_sudachipy+gemma_e4b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 38 | 0 | 0.000 | scipy.stats.binomtest |

### general

| Method A | Method B | A fail/B success | A success/B fail | p | Method |
|---|---|---:|---:|---:|---|
| sudachipy | fugashi_unidic | 0 | 1 | 1.000 | scipy.stats.binomtest |
| sudachipy | gemma_e2b_saved | 3 | 9 | 0.146 | scipy.stats.binomtest |
| sudachipy | gemma_e4b_saved | 4 | 3 | 1.000 | scipy.stats.binomtest |
| sudachipy | hybrid_sudachipy+gemma_e2b_saved | 2 | 9 | 0.065 | scipy.stats.binomtest |
| sudachipy | hybrid_sudachipy+gemma_e4b_saved | 3 | 3 | 1.000 | scipy.stats.binomtest |
| sudachipy | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 5 | 0 | 0.062 | scipy.stats.binomtest |
| fugashi_unidic | gemma_e2b_saved | 3 | 8 | 0.227 | scipy.stats.binomtest |
| fugashi_unidic | gemma_e4b_saved | 5 | 3 | 0.727 | scipy.stats.binomtest |
| fugashi_unidic | hybrid_sudachipy+gemma_e2b_saved | 2 | 8 | 0.109 | scipy.stats.binomtest |
| fugashi_unidic | hybrid_sudachipy+gemma_e4b_saved | 4 | 3 | 1.000 | scipy.stats.binomtest |
| fugashi_unidic | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 6 | 0 | 0.031 | scipy.stats.binomtest |
| gemma_e2b_saved | gemma_e4b_saved | 7 | 0 | 0.016 | scipy.stats.binomtest |
| gemma_e2b_saved | hybrid_sudachipy+gemma_e2b_saved | 0 | 1 | 1.000 | scipy.stats.binomtest |
| gemma_e2b_saved | hybrid_sudachipy+gemma_e4b_saved | 7 | 1 | 0.070 | scipy.stats.binomtest |
| gemma_e2b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 11 | 0 | 0.001 | scipy.stats.binomtest |
| gemma_e4b_saved | hybrid_sudachipy+gemma_e2b_saved | 0 | 8 | 0.008 | scipy.stats.binomtest |
| gemma_e4b_saved | hybrid_sudachipy+gemma_e4b_saved | 0 | 1 | 1.000 | scipy.stats.binomtest |
| gemma_e4b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 4 | 0 | 0.125 | scipy.stats.binomtest |
| hybrid_sudachipy+gemma_e2b_saved | hybrid_sudachipy+gemma_e4b_saved | 7 | 0 | 0.016 | scipy.stats.binomtest |
| hybrid_sudachipy+gemma_e2b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 12 | 0 | 0.000 | scipy.stats.binomtest |
| hybrid_sudachipy+gemma_e4b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 5 | 0 | 0.062 | scipy.stats.binomtest |

## Wilcoxon Character-Distance Tests

### medical

| Method A | Method B | n | Nonzero | statistic | p | Method |
|---|---|---:|---:|---:|---:|---|
| sudachipy | fugashi_unidic | 15 | 1 | 0.000 | 0.317 | scipy.stats.wilcoxon |
| sudachipy | gemma_e2b_saved | 15 | 3 | 2.500 | 0.785 | scipy.stats.wilcoxon |
| sudachipy | gemma_e4b_saved | 15 | 2 | 0.000 | 0.180 | scipy.stats.wilcoxon |
| sudachipy | hybrid_sudachipy+gemma_e2b_saved | 15 | 2 | 0.000 | 0.180 | scipy.stats.wilcoxon |
| sudachipy | hybrid_sudachipy+gemma_e4b_saved | 15 | 1 | 0.000 | 0.317 | scipy.stats.wilcoxon |
| sudachipy | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 15 | 2 | 0.000 | 0.180 | scipy.stats.wilcoxon |
| fugashi_unidic | gemma_e2b_saved | 15 | 3 | 1.000 | 0.285 | scipy.stats.wilcoxon |
| fugashi_unidic | gemma_e4b_saved | 15 | 3 | 0.000 | 0.109 | scipy.stats.wilcoxon |
| fugashi_unidic | hybrid_sudachipy+gemma_e2b_saved | 15 | 2 | 1.000 | 0.655 | scipy.stats.wilcoxon |
| fugashi_unidic | hybrid_sudachipy+gemma_e4b_saved | 15 | 2 | 0.000 | 0.180 | scipy.stats.wilcoxon |
| fugashi_unidic | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 15 | 3 | 0.000 | 0.109 | scipy.stats.wilcoxon |
| gemma_e2b_saved | gemma_e4b_saved | 15 | 3 | 0.000 | 0.102 | scipy.stats.wilcoxon |
| gemma_e2b_saved | hybrid_sudachipy+gemma_e2b_saved | 15 | 1 | 0.000 | 0.317 | scipy.stats.wilcoxon |
| gemma_e2b_saved | hybrid_sudachipy+gemma_e4b_saved | 15 | 4 | 3.500 | 0.577 | scipy.stats.wilcoxon |
| gemma_e2b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 15 | 3 | 0.000 | 0.102 | scipy.stats.wilcoxon |
| gemma_e4b_saved | hybrid_sudachipy+gemma_e2b_saved | 15 | 4 | 0.000 | 0.063 | scipy.stats.wilcoxon |
| gemma_e4b_saved | hybrid_sudachipy+gemma_e4b_saved | 15 | 1 | 0.000 | 0.317 | scipy.stats.wilcoxon |
| gemma_e4b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 15 | 0 | 0.000 | 1.000 | no_difference |
| hybrid_sudachipy+gemma_e2b_saved | hybrid_sudachipy+gemma_e4b_saved | 15 | 3 | 0.000 | 0.102 | scipy.stats.wilcoxon |
| hybrid_sudachipy+gemma_e2b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 15 | 4 | 0.000 | 0.063 | scipy.stats.wilcoxon |
| hybrid_sudachipy+gemma_e4b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 15 | 1 | 0.000 | 0.317 | scipy.stats.wilcoxon |

### sewage

| Method A | Method B | n | Nonzero | statistic | p | Method |
|---|---|---:|---:|---:|---:|---|
| sudachipy | fugashi_unidic | 60 | 1 | 0.000 | 0.317 | scipy.stats.wilcoxon |
| sudachipy | gemma_e2b_saved | 60 | 55 | 21.000 | 0.000 | scipy.stats.wilcoxon |
| sudachipy | gemma_e4b_saved | 60 | 49 | 10.000 | 0.000 | scipy.stats.wilcoxon |
| sudachipy | hybrid_sudachipy+gemma_e2b_saved | 60 | 49 | 19.000 | 0.000 | scipy.stats.wilcoxon |
| sudachipy | hybrid_sudachipy+gemma_e4b_saved | 60 | 43 | 8.000 | 0.000 | scipy.stats.wilcoxon |
| sudachipy | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 60 | 3 | 0.000 | 0.102 | scipy.stats.wilcoxon |
| fugashi_unidic | gemma_e2b_saved | 60 | 54 | 19.000 | 0.000 | scipy.stats.wilcoxon |
| fugashi_unidic | gemma_e4b_saved | 60 | 48 | 8.000 | 0.000 | scipy.stats.wilcoxon |
| fugashi_unidic | hybrid_sudachipy+gemma_e2b_saved | 60 | 50 | 28.500 | 0.000 | scipy.stats.wilcoxon |
| fugashi_unidic | hybrid_sudachipy+gemma_e4b_saved | 60 | 44 | 13.000 | 0.000 | scipy.stats.wilcoxon |
| fugashi_unidic | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 60 | 4 | 0.000 | 0.066 | scipy.stats.wilcoxon |
| gemma_e2b_saved | gemma_e4b_saved | 60 | 7 | 2.500 | 0.047 | scipy.stats.wilcoxon |
| gemma_e2b_saved | hybrid_sudachipy+gemma_e2b_saved | 60 | 6 | 0.000 | 0.020 | scipy.stats.wilcoxon |
| gemma_e2b_saved | hybrid_sudachipy+gemma_e4b_saved | 60 | 13 | 2.500 | 0.002 | scipy.stats.wilcoxon |
| gemma_e2b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 60 | 53 | 0.000 | 0.000 | scipy.stats.wilcoxon |
| gemma_e4b_saved | hybrid_sudachipy+gemma_e2b_saved | 60 | 13 | 32.500 | 0.359 | scipy.stats.wilcoxon |
| gemma_e4b_saved | hybrid_sudachipy+gemma_e4b_saved | 60 | 6 | 0.000 | 0.020 | scipy.stats.wilcoxon |
| gemma_e4b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 60 | 48 | 0.000 | 0.000 | scipy.stats.wilcoxon |
| hybrid_sudachipy+gemma_e2b_saved | hybrid_sudachipy+gemma_e4b_saved | 60 | 7 | 2.500 | 0.047 | scipy.stats.wilcoxon |
| hybrid_sudachipy+gemma_e2b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 60 | 47 | 0.000 | 0.000 | scipy.stats.wilcoxon |
| hybrid_sudachipy+gemma_e4b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 60 | 42 | 0.000 | 0.000 | scipy.stats.wilcoxon |

### general

| Method A | Method B | n | Nonzero | statistic | p | Method |
|---|---|---:|---:|---:|---:|---|
| sudachipy | fugashi_unidic | 30 | 1 | 0.000 | 0.317 | scipy.stats.wilcoxon |
| sudachipy | gemma_e2b_saved | 30 | 14 | 34.500 | 0.254 | scipy.stats.wilcoxon |
| sudachipy | gemma_e4b_saved | 30 | 7 | 8.500 | 0.343 | scipy.stats.wilcoxon |
| sudachipy | hybrid_sudachipy+gemma_e2b_saved | 30 | 13 | 22.000 | 0.097 | scipy.stats.wilcoxon |
| sudachipy | hybrid_sudachipy+gemma_e4b_saved | 30 | 6 | 8.000 | 0.595 | scipy.stats.wilcoxon |
| sudachipy | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 30 | 5 | 0.000 | 0.039 | scipy.stats.wilcoxon |
| fugashi_unidic | gemma_e2b_saved | 30 | 14 | 47.000 | 0.728 | scipy.stats.wilcoxon |
| fugashi_unidic | gemma_e4b_saved | 30 | 8 | 8.500 | 0.177 | scipy.stats.wilcoxon |
| fugashi_unidic | hybrid_sudachipy+gemma_e2b_saved | 30 | 13 | 34.000 | 0.417 | scipy.stats.wilcoxon |
| fugashi_unidic | hybrid_sudachipy+gemma_e4b_saved | 30 | 7 | 8.000 | 0.306 | scipy.stats.wilcoxon |
| fugashi_unidic | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 30 | 6 | 0.000 | 0.026 | scipy.stats.wilcoxon |
| gemma_e2b_saved | gemma_e4b_saved | 30 | 10 | 2.500 | 0.010 | scipy.stats.wilcoxon |
| gemma_e2b_saved | hybrid_sudachipy+gemma_e2b_saved | 30 | 1 | 0.000 | 0.317 | scipy.stats.wilcoxon |
| gemma_e2b_saved | hybrid_sudachipy+gemma_e4b_saved | 30 | 11 | 12.000 | 0.059 | scipy.stats.wilcoxon |
| gemma_e2b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 30 | 12 | 0.000 | 0.002 | scipy.stats.wilcoxon |
| gemma_e4b_saved | hybrid_sudachipy+gemma_e2b_saved | 30 | 11 | 2.500 | 0.006 | scipy.stats.wilcoxon |
| gemma_e4b_saved | hybrid_sudachipy+gemma_e4b_saved | 30 | 1 | 0.000 | 0.317 | scipy.stats.wilcoxon |
| gemma_e4b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 30 | 4 | 0.000 | 0.059 | scipy.stats.wilcoxon |
| hybrid_sudachipy+gemma_e2b_saved | hybrid_sudachipy+gemma_e4b_saved | 30 | 10 | 2.500 | 0.010 | scipy.stats.wilcoxon |
| hybrid_sudachipy+gemma_e2b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 30 | 13 | 0.000 | 0.001 | scipy.stats.wilcoxon |
| hybrid_sudachipy+gemma_e4b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 30 | 5 | 0.000 | 0.038 | scipy.stats.wilcoxon |

## Error Analysis

| Method | Domain | Error Type | Count |
|---|---|---|---:|
| sudachipy | medical | 部分誤読 | 2 |
| sudachipy | sewage | 部分誤読 | 9 |
| sudachipy | sewage | 完全誤読 | 1 |
| sudachipy | general | 部分誤読 | 6 |
| fugashi_unidic | medical | 部分誤読 | 2 |
| fugashi_unidic | medical | 完全誤読 | 1 |
| fugashi_unidic | sewage | 部分誤読 | 8 |
| fugashi_unidic | sewage | 完全誤読 | 2 |
| fugashi_unidic | general | 部分誤読 | 6 |
| fugashi_unidic | general | 完全誤読 | 1 |
| gemma_e2b_saved | medical | 部分誤読 | 3 |
| gemma_e2b_saved | sewage | 部分誤読 | 9 |
| gemma_e2b_saved | sewage | 完全誤読 | 45 |
| gemma_e2b_saved | general | 部分誤読 | 12 |
| gemma_e4b_saved | sewage | 部分誤読 | 5 |
| gemma_e4b_saved | sewage | 完全誤読 | 45 |
| gemma_e4b_saved | general | 部分誤読 | 5 |
| hybrid_sudachipy+gemma_e2b_saved | medical | 部分誤読 | 4 |
| hybrid_sudachipy+gemma_e2b_saved | sewage | 部分誤読 | 10 |
| hybrid_sudachipy+gemma_e2b_saved | sewage | 完全誤読 | 39 |
| hybrid_sudachipy+gemma_e2b_saved | general | 部分誤読 | 13 |
| hybrid_sudachipy+gemma_e4b_saved | medical | 部分誤読 | 1 |
| hybrid_sudachipy+gemma_e4b_saved | sewage | 部分誤読 | 6 |
| hybrid_sudachipy+gemma_e4b_saved | sewage | 完全誤読 | 39 |
| hybrid_sudachipy+gemma_e4b_saved | general | 部分誤読 | 6 |
| smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | sewage | 部分誤読 | 6 |
| smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | sewage | 完全誤読 | 1 |
| smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | general | 部分誤読 | 1 |

## Complementarity

### medical

| Method A | Method B | A fail/B success | A success/B fail | Either succeeds | Both fail | Rate |
|---|---|---:|---:|---:|---:|---:|
| sudachipy | fugashi_unidic | 0 | 1 | 13 | 2 | 0.067 |
| sudachipy | gemma_e2b_saved | 1 | 2 | 14 | 1 | 0.200 |
| sudachipy | gemma_e4b_saved | 2 | 0 | 15 | 0 | 0.133 |
| sudachipy | hybrid_sudachipy+gemma_e2b_saved | 0 | 2 | 13 | 2 | 0.133 |
| sudachipy | hybrid_sudachipy+gemma_e4b_saved | 1 | 0 | 14 | 1 | 0.067 |
| sudachipy | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 2 | 0 | 15 | 0 | 0.133 |
| fugashi_unidic | gemma_e2b_saved | 1 | 1 | 13 | 2 | 0.133 |
| fugashi_unidic | gemma_e4b_saved | 3 | 0 | 15 | 0 | 0.200 |
| fugashi_unidic | hybrid_sudachipy+gemma_e2b_saved | 0 | 1 | 12 | 3 | 0.067 |
| fugashi_unidic | hybrid_sudachipy+gemma_e4b_saved | 2 | 0 | 14 | 1 | 0.133 |
| fugashi_unidic | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 3 | 0 | 15 | 0 | 0.200 |
| gemma_e2b_saved | gemma_e4b_saved | 3 | 0 | 15 | 0 | 0.200 |
| gemma_e2b_saved | hybrid_sudachipy+gemma_e2b_saved | 0 | 1 | 12 | 3 | 0.067 |
| gemma_e2b_saved | hybrid_sudachipy+gemma_e4b_saved | 3 | 1 | 15 | 0 | 0.267 |
| gemma_e2b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 3 | 0 | 15 | 0 | 0.200 |
| gemma_e4b_saved | hybrid_sudachipy+gemma_e2b_saved | 0 | 4 | 15 | 0 | 0.267 |
| gemma_e4b_saved | hybrid_sudachipy+gemma_e4b_saved | 0 | 1 | 15 | 0 | 0.067 |
| gemma_e4b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 0 | 0 | 15 | 0 | 0.000 |
| hybrid_sudachipy+gemma_e2b_saved | hybrid_sudachipy+gemma_e4b_saved | 3 | 0 | 14 | 1 | 0.200 |
| hybrid_sudachipy+gemma_e2b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 4 | 0 | 15 | 0 | 0.267 |
| hybrid_sudachipy+gemma_e4b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 1 | 0 | 15 | 0 | 0.067 |

### sewage

| Method A | Method B | A fail/B success | A success/B fail | Either succeeds | Both fail | Rate |
|---|---|---:|---:|---:|---:|---:|
| sudachipy | fugashi_unidic | 0 | 0 | 50 | 10 | 0.000 |
| sudachipy | gemma_e2b_saved | 2 | 46 | 52 | 8 | 0.800 |
| sudachipy | gemma_e4b_saved | 2 | 42 | 52 | 8 | 0.733 |
| sudachipy | hybrid_sudachipy+gemma_e2b_saved | 2 | 41 | 52 | 8 | 0.717 |
| sudachipy | hybrid_sudachipy+gemma_e4b_saved | 2 | 37 | 52 | 8 | 0.650 |
| sudachipy | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 3 | 0 | 53 | 7 | 0.050 |
| fugashi_unidic | gemma_e2b_saved | 2 | 46 | 52 | 8 | 0.800 |
| fugashi_unidic | gemma_e4b_saved | 2 | 42 | 52 | 8 | 0.733 |
| fugashi_unidic | hybrid_sudachipy+gemma_e2b_saved | 2 | 41 | 52 | 8 | 0.717 |
| fugashi_unidic | hybrid_sudachipy+gemma_e4b_saved | 2 | 37 | 52 | 8 | 0.650 |
| fugashi_unidic | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 3 | 0 | 53 | 7 | 0.050 |
| gemma_e2b_saved | gemma_e4b_saved | 4 | 0 | 10 | 50 | 0.067 |
| gemma_e2b_saved | hybrid_sudachipy+gemma_e2b_saved | 5 | 0 | 11 | 49 | 0.083 |
| gemma_e2b_saved | hybrid_sudachipy+gemma_e4b_saved | 9 | 0 | 15 | 45 | 0.150 |
| gemma_e2b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 47 | 0 | 53 | 7 | 0.783 |
| gemma_e4b_saved | hybrid_sudachipy+gemma_e2b_saved | 5 | 4 | 15 | 45 | 0.150 |
| gemma_e4b_saved | hybrid_sudachipy+gemma_e4b_saved | 5 | 0 | 15 | 45 | 0.083 |
| gemma_e4b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 43 | 0 | 53 | 7 | 0.717 |
| hybrid_sudachipy+gemma_e2b_saved | hybrid_sudachipy+gemma_e4b_saved | 4 | 0 | 15 | 45 | 0.067 |
| hybrid_sudachipy+gemma_e2b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 42 | 0 | 53 | 7 | 0.700 |
| hybrid_sudachipy+gemma_e4b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 38 | 0 | 53 | 7 | 0.633 |

### general

| Method A | Method B | A fail/B success | A success/B fail | Either succeeds | Both fail | Rate |
|---|---|---:|---:|---:|---:|---:|
| sudachipy | fugashi_unidic | 0 | 1 | 24 | 6 | 0.033 |
| sudachipy | gemma_e2b_saved | 3 | 9 | 27 | 3 | 0.400 |
| sudachipy | gemma_e4b_saved | 4 | 3 | 28 | 2 | 0.233 |
| sudachipy | hybrid_sudachipy+gemma_e2b_saved | 2 | 9 | 26 | 4 | 0.367 |
| sudachipy | hybrid_sudachipy+gemma_e4b_saved | 3 | 3 | 27 | 3 | 0.200 |
| sudachipy | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 5 | 0 | 29 | 1 | 0.167 |
| fugashi_unidic | gemma_e2b_saved | 3 | 8 | 26 | 4 | 0.367 |
| fugashi_unidic | gemma_e4b_saved | 5 | 3 | 28 | 2 | 0.267 |
| fugashi_unidic | hybrid_sudachipy+gemma_e2b_saved | 2 | 8 | 25 | 5 | 0.333 |
| fugashi_unidic | hybrid_sudachipy+gemma_e4b_saved | 4 | 3 | 27 | 3 | 0.233 |
| fugashi_unidic | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 6 | 0 | 29 | 1 | 0.200 |
| gemma_e2b_saved | gemma_e4b_saved | 7 | 0 | 25 | 5 | 0.233 |
| gemma_e2b_saved | hybrid_sudachipy+gemma_e2b_saved | 0 | 1 | 18 | 12 | 0.033 |
| gemma_e2b_saved | hybrid_sudachipy+gemma_e4b_saved | 7 | 1 | 25 | 5 | 0.267 |
| gemma_e2b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 11 | 0 | 29 | 1 | 0.367 |
| gemma_e4b_saved | hybrid_sudachipy+gemma_e2b_saved | 0 | 8 | 25 | 5 | 0.267 |
| gemma_e4b_saved | hybrid_sudachipy+gemma_e4b_saved | 0 | 1 | 25 | 5 | 0.033 |
| gemma_e4b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 4 | 0 | 29 | 1 | 0.133 |
| hybrid_sudachipy+gemma_e2b_saved | hybrid_sudachipy+gemma_e4b_saved | 7 | 0 | 24 | 6 | 0.233 |
| hybrid_sudachipy+gemma_e2b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 12 | 0 | 29 | 1 | 0.400 |
| hybrid_sudachipy+gemma_e4b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 5 | 0 | 29 | 1 | 0.167 |

## Oracle Upper Bounds

### medical

| Method A | Method B | n | Oracle exact | Oracle accuracy |
|---|---|---:|---:|---:|
| sudachipy | fugashi_unidic | 15 | 13 | 0.867 |
| sudachipy | gemma_e2b_saved | 15 | 14 | 0.933 |
| sudachipy | gemma_e4b_saved | 15 | 15 | 1.000 |
| sudachipy | hybrid_sudachipy+gemma_e2b_saved | 15 | 13 | 0.867 |
| sudachipy | hybrid_sudachipy+gemma_e4b_saved | 15 | 14 | 0.933 |
| sudachipy | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 15 | 15 | 1.000 |
| fugashi_unidic | gemma_e2b_saved | 15 | 13 | 0.867 |
| fugashi_unidic | gemma_e4b_saved | 15 | 15 | 1.000 |
| fugashi_unidic | hybrid_sudachipy+gemma_e2b_saved | 15 | 12 | 0.800 |
| fugashi_unidic | hybrid_sudachipy+gemma_e4b_saved | 15 | 14 | 0.933 |
| fugashi_unidic | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 15 | 15 | 1.000 |
| gemma_e2b_saved | gemma_e4b_saved | 15 | 15 | 1.000 |
| gemma_e2b_saved | hybrid_sudachipy+gemma_e2b_saved | 15 | 12 | 0.800 |
| gemma_e2b_saved | hybrid_sudachipy+gemma_e4b_saved | 15 | 15 | 1.000 |
| gemma_e2b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 15 | 15 | 1.000 |
| gemma_e4b_saved | hybrid_sudachipy+gemma_e2b_saved | 15 | 15 | 1.000 |
| gemma_e4b_saved | hybrid_sudachipy+gemma_e4b_saved | 15 | 15 | 1.000 |
| gemma_e4b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 15 | 15 | 1.000 |
| hybrid_sudachipy+gemma_e2b_saved | hybrid_sudachipy+gemma_e4b_saved | 15 | 14 | 0.933 |
| hybrid_sudachipy+gemma_e2b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 15 | 15 | 1.000 |
| hybrid_sudachipy+gemma_e4b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 15 | 15 | 1.000 |

### sewage

| Method A | Method B | n | Oracle exact | Oracle accuracy |
|---|---|---:|---:|---:|
| sudachipy | fugashi_unidic | 60 | 50 | 0.833 |
| sudachipy | gemma_e2b_saved | 60 | 52 | 0.867 |
| sudachipy | gemma_e4b_saved | 60 | 52 | 0.867 |
| sudachipy | hybrid_sudachipy+gemma_e2b_saved | 60 | 52 | 0.867 |
| sudachipy | hybrid_sudachipy+gemma_e4b_saved | 60 | 52 | 0.867 |
| sudachipy | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 60 | 53 | 0.883 |
| fugashi_unidic | gemma_e2b_saved | 60 | 52 | 0.867 |
| fugashi_unidic | gemma_e4b_saved | 60 | 52 | 0.867 |
| fugashi_unidic | hybrid_sudachipy+gemma_e2b_saved | 60 | 52 | 0.867 |
| fugashi_unidic | hybrid_sudachipy+gemma_e4b_saved | 60 | 52 | 0.867 |
| fugashi_unidic | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 60 | 53 | 0.883 |
| gemma_e2b_saved | gemma_e4b_saved | 60 | 10 | 0.167 |
| gemma_e2b_saved | hybrid_sudachipy+gemma_e2b_saved | 60 | 11 | 0.183 |
| gemma_e2b_saved | hybrid_sudachipy+gemma_e4b_saved | 60 | 15 | 0.250 |
| gemma_e2b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 60 | 53 | 0.883 |
| gemma_e4b_saved | hybrid_sudachipy+gemma_e2b_saved | 60 | 15 | 0.250 |
| gemma_e4b_saved | hybrid_sudachipy+gemma_e4b_saved | 60 | 15 | 0.250 |
| gemma_e4b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 60 | 53 | 0.883 |
| hybrid_sudachipy+gemma_e2b_saved | hybrid_sudachipy+gemma_e4b_saved | 60 | 15 | 0.250 |
| hybrid_sudachipy+gemma_e2b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 60 | 53 | 0.883 |
| hybrid_sudachipy+gemma_e4b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 60 | 53 | 0.883 |

### general

| Method A | Method B | n | Oracle exact | Oracle accuracy |
|---|---|---:|---:|---:|
| sudachipy | fugashi_unidic | 30 | 24 | 0.800 |
| sudachipy | gemma_e2b_saved | 30 | 27 | 0.900 |
| sudachipy | gemma_e4b_saved | 30 | 28 | 0.933 |
| sudachipy | hybrid_sudachipy+gemma_e2b_saved | 30 | 26 | 0.867 |
| sudachipy | hybrid_sudachipy+gemma_e4b_saved | 30 | 27 | 0.900 |
| sudachipy | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 30 | 29 | 0.967 |
| fugashi_unidic | gemma_e2b_saved | 30 | 26 | 0.867 |
| fugashi_unidic | gemma_e4b_saved | 30 | 28 | 0.933 |
| fugashi_unidic | hybrid_sudachipy+gemma_e2b_saved | 30 | 25 | 0.833 |
| fugashi_unidic | hybrid_sudachipy+gemma_e4b_saved | 30 | 27 | 0.900 |
| fugashi_unidic | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 30 | 29 | 0.967 |
| gemma_e2b_saved | gemma_e4b_saved | 30 | 25 | 0.833 |
| gemma_e2b_saved | hybrid_sudachipy+gemma_e2b_saved | 30 | 18 | 0.600 |
| gemma_e2b_saved | hybrid_sudachipy+gemma_e4b_saved | 30 | 25 | 0.833 |
| gemma_e2b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 30 | 29 | 0.967 |
| gemma_e4b_saved | hybrid_sudachipy+gemma_e2b_saved | 30 | 25 | 0.833 |
| gemma_e4b_saved | hybrid_sudachipy+gemma_e4b_saved | 30 | 25 | 0.833 |
| gemma_e4b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 30 | 29 | 0.967 |
| hybrid_sudachipy+gemma_e2b_saved | hybrid_sudachipy+gemma_e4b_saved | 30 | 24 | 0.800 |
| hybrid_sudachipy+gemma_e2b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 30 | 29 | 0.967 |
| hybrid_sudachipy+gemma_e4b_saved | smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved | 30 | 29 | 0.967 |

## Key Findings

- medical: best observed accuracy is 1.000 from gemma_e4b_saved.
- sewage: best observed accuracy is 0.883 from smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved.
- general: best observed accuracy is 0.967 from smart_hybrid_sudachipy_fugashi_unidic+gemma_e4b_saved.
- Dictionary rows appear only for dependencies that installed successfully in this environment; skipped rows are not imputed.
