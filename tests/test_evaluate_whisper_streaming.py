from pathlib import Path

import pytest
import yaml

from scripts.evaluate_whisper_streaming import (
    edit_operation_counts,
    filter_supported_kwargs,
    ngram_repetition_stats,
    optional_positive_int,
    parse_temperature,
    should_skip_no_speech_segment,
)


def test_signature_filter_drops_options_missing_from_older_faster_whisper():
    def old_transcribe(audio, *, language=None, beam_size=5):  # noqa: ANN001
        return audio, language, beam_size

    filtered, unsupported = filter_supported_kwargs(
        old_transcribe,
        {
            "language": "ja",
            "beam_size": 3,
            "repetition_penalty": 1.1,
            "hallucination_silence_threshold": 2.0,
            "max_new_tokens": 192,
        },
    )

    assert filtered == {"language": "ja", "beam_size": 3}
    assert unsupported == [
        "hallucination_silence_threshold",
        "max_new_tokens",
        "repetition_penalty",
    ]


def test_signature_filter_keeps_all_options_for_kwargs_api():
    def transcribe(audio, **kwargs):  # noqa: ANN001
        return audio, kwargs

    options = {"temperature": 0.0, "no_repeat_ngram_size": 3}
    assert filter_supported_kwargs(transcribe, options) == (options, [])


def test_edit_operation_counts_separates_insertions_from_other_errors():
    assert edit_operation_counts(list("配管"), list("配水管")) == {
        "substitutions": 0,
        "deletions": 0,
        "insertions": 1,
    }
    assert edit_operation_counts([], list("湧出し"))["insertions"] == 3


def test_repetition_stats_count_second_and_later_ngram_occurrences():
    stats = ngram_repetition_stats(list("abcabcabc"), ngram_size=3)
    assert stats["total_ngrams"] == 7
    assert stats["repeated_ngrams"] == 4
    assert stats["repetition_ratio"] == 4 / 7


def test_temperature_accepts_scalar_and_fallback_sequence():
    assert parse_temperature(0) == 0.0
    assert parse_temperature([0, 0.2]) == [0.0, 0.2]


def test_max_new_tokens_accepts_null_but_rejects_non_positive_caps():
    assert optional_positive_int(None, name="max_new_tokens") is None
    assert optional_positive_int("192", name="max_new_tokens") == 192
    with pytest.raises(ValueError, match="positive"):
        optional_positive_int(0, name="max_new_tokens")


def test_no_speech_filter_requires_both_silence_probability_and_low_logprob():
    options = {"no_speech_threshold": 0.6, "log_prob_threshold": -1.0}
    assert should_skip_no_speech_segment(0.8, -1.2, **options)
    assert not should_skip_no_speech_segment(0.8, -0.2, **options)
    assert not should_skip_no_speech_segment(0.2, -2.0, **options)


def test_streaming_config_uses_anti_hallucination_defaults():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "whisper_streaming_eval.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    streaming = config["whisper_streaming"]
    model = config["model"]

    assert streaming["vad"] is True
    assert streaming["condition_on_previous_text"] is False
    assert streaming["use_initial_prompt"] is False
    assert streaming["temperature"] == 0.0
    assert streaming["repetition_penalty"] >= 1.0
    assert streaming["no_repeat_ngram_size"] == 3
    assert streaming["max_new_tokens"] == 192
    assert model["beam_size"] == 3
