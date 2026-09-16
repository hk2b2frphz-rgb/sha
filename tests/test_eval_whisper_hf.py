import json
from pathlib import Path

import pytest

from eval.asr_text import error_rate, normalize_text, term_hit
from scripts.eval_whisper_hf import load_manifest


def test_load_manifest_accepts_both_repository_schemas(tmp_path: Path):
    path = tmp_path / "m.jsonl"
    path.write_text(
        json.dumps({"id": "1", "audio": "/a/1.wav", "text": "心電図の所見"}, ensure_ascii=False)
        + "\n"
        + json.dumps({"id": "2", "wav": "/a/2.wav", "sentence": "配管の点検", "term": "配管"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    rows = load_manifest(path)
    assert [r["audio"] for r in rows] == ["/a/1.wav", "/a/2.wav"]
    assert [r["reference"] for r in rows] == ["心電図の所見", "配管の点検"]
    assert rows[1]["terms"] == ["配管"]
    assert rows[0]["terms"] == []


def test_load_manifest_limit_and_empty_handling(tmp_path: Path):
    path = tmp_path / "m.jsonl"
    path.write_text(
        "".join(
            json.dumps({"id": str(i), "audio": f"/a/{i}.wav", "text": "文"}) + "\n" for i in range(5)
        ),
        encoding="utf-8",
    )
    assert len(load_manifest(path, limit=2)) == 2

    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_manifest(empty)


def test_load_manifest_rejects_a_row_without_audio(tmp_path: Path):
    path = tmp_path / "m.jsonl"
    path.write_text(json.dumps({"id": "1", "text": "文"}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_manifest(path)


def test_metrics_ignore_punctuation_and_case():
    assert normalize_text("これは、テストです。") == normalize_text("これはテストです")
    assert error_rate(list("あいうえお"), list("あいうえお")) == 0.0
    assert error_rate(list("あいうえお"), list("あいうえか")) == pytest.approx(0.2)
    assert error_rate([], []) == 0.0
    assert error_rate([], ["a"]) == 1.0


def test_term_hit_matches_through_punctuation():
    assert term_hit("流量計", "現場の、流量計を点検した。")
    assert not term_hit("流量計", "現場の計器を点検した")
    assert not term_hit("", "何か")
