from scripts.prepare_asr_eval_jsonl import load_readings


def test_load_readings_accepts_comma_delimited_mapping(tmp_path):
    readings_file = tmp_path / "readings.tsv"
    readings_file.write_text("term,reading\n心筋梗塞,しんきんこうそく\n", encoding="utf-8")

    assert load_readings(readings_file) == [("心筋梗塞", "しんきんこうそく")]


def test_load_readings_accepts_tab_delimited_mapping(tmp_path):
    readings_file = tmp_path / "readings.tsv"
    readings_file.write_text("term\treading\n心筋梗塞\tしんきんこうそく\n", encoding="utf-8")

    assert load_readings(readings_file) == [("心筋梗塞", "しんきんこうそく")]
