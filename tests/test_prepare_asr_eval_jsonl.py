from scripts.prepare_asr_eval_jsonl import apply_readings


def test_apply_readings_replaces_only_matching_technical_terms():
    sentence = "心筋梗塞の患者を診察する"
    result = apply_readings(sentence, [("心筋梗塞", "しんきんこうそく")])
    assert result == "しんきんこうそくの患者を診察する"
