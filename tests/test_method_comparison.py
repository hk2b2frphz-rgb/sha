from argparse import Namespace

from experiments.run_method_comparison import aggregate_evaluation, bootstrap_accuracy_ci, mcnemar_test


def test_bootstrap_accuracy_ci_is_deterministic_and_bounded():
    ci = bootstrap_accuracy_ci([True, True, False, False], n_resamples=200, seed=7)
    assert 0.0 <= ci["lower"] <= ci["upper"] <= 1.0
    assert ci == bootstrap_accuracy_ci([True, True, False, False], n_resamples=200, seed=7)


def test_mcnemar_counts_discordant_pairs():
    result = mcnemar_test([True, False, False, True], [False, True, False, True])
    assert result["b01"] == 1
    assert result["b10"] == 1
    assert result["chi2"] == 0.5
    assert result["p_value"] == 0.4795


def test_aggregate_evaluation_computes_character_distance_average():
    evaluation = {
        "results": [
            {"exact_match": True, "character_distance": 0},
            {"exact_match": False, "character_distance": 2},
            {"status": "missing_gold"},
        ],
        "exact_match_accuracy": 0.5,
        "average_mora_error_rate": 0.25,
        "n_exact_match": 1,
    }
    row = aggregate_evaluation(evaluation)
    assert row["n"] == 2
    assert row["average_character_distance"] == 1.0
