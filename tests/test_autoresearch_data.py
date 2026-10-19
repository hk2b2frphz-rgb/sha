import json
from pathlib import Path

from autoresearch.data import collect_terms, read_jsonl, split_manifest


def write_manifest(path: Path, n: int) -> None:
    path.write_text(
        "".join(
            json.dumps({"id": f"{i:04d}", "wav": f"/audio/{i}.wav", "sentence": "文", "term": f"用語{i % 3}"},
                       ensure_ascii=False)
            + "\n"
            for i in range(n)
        ),
        encoding="utf-8",
    )


def test_split_is_disjoint_covers_everything_and_is_reproducible(tmp_path: Path):
    source = tmp_path / "test_manifest.jsonl"
    write_manifest(source, 40)
    dev, holdout = tmp_path / "dev.jsonl", tmp_path / "holdout.jsonl"

    n_dev, n_hold = split_manifest(source, dev, holdout, dev_ratio=0.5, seed=42)
    assert n_dev + n_hold == 40
    dev_ids = {r["id"] for r in read_jsonl(dev)}
    holdout_ids = {r["id"] for r in read_jsonl(holdout)}
    assert dev_ids.isdisjoint(holdout_ids)
    assert len(dev_ids | holdout_ids) == 40

    # A resumed job must land on exactly the same split.
    split_manifest(source, tmp_path / "dev2.jsonl", tmp_path / "hold2.jsonl", dev_ratio=0.5, seed=42)
    assert {r["id"] for r in read_jsonl(tmp_path / "dev2.jsonl")} == dev_ids


def test_dev_ratio_shifts_the_balance(tmp_path: Path):
    source = tmp_path / "m.jsonl"
    write_manifest(source, 100)
    n_dev, n_hold = split_manifest(source, tmp_path / "d.jsonl", tmp_path / "h.jsonl", dev_ratio=0.2)
    assert n_dev < n_hold
    assert n_dev + n_hold == 100


def test_dev_split_is_never_empty(tmp_path: Path):
    source = tmp_path / "tiny.jsonl"
    write_manifest(source, 2)
    n_dev, _ = split_manifest(source, tmp_path / "d.jsonl", tmp_path / "h.jsonl", dev_ratio=0.0)
    assert n_dev >= 1


def test_collect_terms_deduplicates_in_order(tmp_path: Path):
    source = tmp_path / "m.jsonl"
    write_manifest(source, 9)
    assert collect_terms(source) == ["用語0", "用語1", "用語2"]


def test_collect_terms_spans_sources_and_skips_missing_files(tmp_path: Path):
    first = tmp_path / "train.jsonl"  # manifest without any term column
    first.write_text(
        json.dumps({"id": "1", "audio": "/a.wav", "text": "文"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    second = tmp_path / "tts_input.jsonl"
    write_manifest(second, 6)
    assert collect_terms(first, tmp_path / "absent.jsonl", second) == ["用語0", "用語1", "用語2"]


def test_collect_terms_returns_empty_when_nothing_is_annotated(tmp_path: Path):
    path = tmp_path / "plain.jsonl"
    path.write_text(json.dumps({"id": "1", "audio": "/a.wav", "text": "文"}) + "\n", encoding="utf-8")
    assert collect_terms(path) == []
