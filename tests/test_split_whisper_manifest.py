import pytest

from scripts.split_whisper_manifest import (
    atomic_write_many,
    build_summary,
    record_identity_keys,
    positive_term,
    split_rows,
)


def make_row(index: int, term: str, *, kind: str = "positive") -> dict[str, str]:
    return {
        "id": f"id-{index}",
        "audio": f"/audio/{index}.wav",
        "text": f"現場で{term}を点検する",
        "kind": kind,
        "source_term": term,
        "term": term,
    }


def test_stratified_split_is_deterministic_and_places_each_positive_term_in_both():
    rows = [make_row(i, "流量計") for i in range(4)]
    # singleton strata に分かれていても term-level repair で双方へ置かれる。
    rows.extend(
        [
            make_row(10, "活性汚泥", kind="generated_a"),
            make_row(11, "活性汚泥", kind="generated_b"),
        ]
    )

    first = split_rows(rows, dev_ratio=0.25, seed=123)
    second = split_rows(list(reversed(rows)), dev_ratio=0.25, seed=123)

    first_train = {row["id"] for row in first.train}
    first_dev = {row["id"] for row in first.dev}
    assert first_train == {row["id"] for row in second.train}
    assert first_dev == {row["id"] for row in second.dev}
    assert first_train.isdisjoint(first_dev)
    for term in ("流量計", "活性汚泥"):
        assert any(row["term"] == term for row in first.train)
        assert any(row["term"] == term for row in first.dev)


def test_exact_duplicates_are_removed_without_cross_split_identity_overlap():
    rows = [make_row(1, "配管"), make_row(1, "配管"), make_row(2, "配管")]
    result = split_rows(rows, dev_ratio=0.5, seed=42)

    assert result.duplicates_removed == 1
    train_keys = {key for row in result.train for key in record_identity_keys(row)}
    dev_keys = {key for row in result.dev for key in record_identity_keys(row)}
    assert train_keys.isdisjoint(dev_keys)


def test_conflicting_duplicate_audio_is_rejected():
    first = make_row(1, "配管")
    second = make_row(2, "配管")
    second["audio"] = first["audio"]

    with pytest.raises(ValueError, match="conflicts"):
        split_rows([first, second], dev_ratio=0.5, seed=42)


def test_replay_and_silence_controls_are_not_counted_as_term_positives():
    replay = make_row(1, "配管", kind="replay")
    silence = make_row(2, "配管", kind="silence")
    silence["text"] = ""

    assert positive_term(replay) is None
    assert positive_term(silence) is None


def test_summary_reports_strata_and_positive_coverage():
    rows = [make_row(i, "沈殿池") for i in range(4)]
    result = split_rows(rows, dev_ratio=0.25, seed=7)
    summary = build_summary(result, input_count=len(rows), dev_ratio=0.25, seed=7)

    assert summary["identity_overlap"] == 0
    assert summary["positive_terms_possible_in_both"] == 1
    assert summary["positive_terms_present_in_both"] == 1
    assert summary["train_records"] + summary["dev_records"] == 4


def test_atomic_write_many_replaces_complete_files(tmp_path):
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    train.write_text("old\n", encoding="utf-8")

    atomic_write_many([(train, "new train\n"), (dev, "new dev\n")])

    assert train.read_text(encoding="utf-8") == "new train\n"
    assert dev.read_text(encoding="utf-8") == "new dev\n"
    assert not list(tmp_path.glob("*.tmp"))
