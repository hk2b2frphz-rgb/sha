import json
import wave
from pathlib import Path

import pytest

from scripts.build_whisper_manifest import build_rows


def write_wav(path: Path, seconds: float = 0.2, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(b"\0\0" * round(seconds * sample_rate))


def write_manifest(directory: Path, rows: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_build_rows_keeps_kanji_target_and_provenance(tmp_path: Path):
    synth = tmp_path / "shard"
    wav = synth / "wav" / "term_1.wav"
    write_wav(wav)
    write_manifest(
        synth,
        [
            {
                "id": "term_1",
                "wav": "wav/term_1.wav",
                "duration_sec": 0.2,
                # This must not become the Whisper label.
                "sentence": "かっせいおでいほうを確認します。",
                "synthesis_text": "活性汚泥法を確認します。",
                "pronunciation_check": "passed",
                "pronunciation_asr_text": "活性汚泥法を確認します。",
                "pronunciation_observed_reading": "かっせいおでいほうをかくにんします",
                "pronunciation_mora_distance": 0.0,
                "pronunciation_content_edits": 0,
                "reading_fallback_used": False,
            }
        ],
    )
    targets = [
        {
            "id": "term_1",
            "kind": "term",
            "term": "活性汚泥法",
            "reading": "かっせいおでいほう",
            "sentence": "活性汚泥法を確認します。",
            "tts_text": "かっせいおでいほうを確認します。",
        }
    ]

    rows, summary = build_rows(targets, [synth], min_duration=0.05, max_duration=2.0)

    assert rows[0]["text"] == "活性汚泥法を確認します。"
    assert rows[0]["tts_text"] == "かっせいおでいほうを確認します。"
    assert rows[0]["synthesis_text"] == "活性汚泥法を確認します。"
    assert rows[0]["pronunciation_check"] == "passed"
    assert rows[0]["pronunciation_mora_distance"] == 0.0
    assert rows[0]["pronunciation_content_edits"] == 0
    assert not rows[0]["reading_fallback_used"]
    assert rows[0]["kind"] == "term"
    assert summary["kind_counts"] == {"term": 1}


def test_build_rows_preserves_empty_label_for_silence_control(tmp_path: Path):
    synth = tmp_path / "shard"
    write_wav(synth / "wav" / "silence.wav", seconds=0.4)
    write_manifest(
        synth,
        [{"id": "silence", "wav": "wav/silence.wav", "duration_sec": 0.4}],
    )
    targets = [{"id": "silence", "kind": "silence", "sentence": "", "tts_text": ""}]

    rows, _summary = build_rows(targets, [synth], min_duration=0.05, max_duration=2.0)

    assert rows[0]["text"] == ""
    assert rows[0]["kind"] == "silence"


def test_build_rows_reports_missing_and_rejects_duplicate_synthesis(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    for directory in (first, second):
        write_wav(directory / "wav" / "x.wav")
        write_manifest(directory, [{"id": "x", "wav": "wav/x.wav", "duration_sec": 0.2}])

    with pytest.raises(SystemExit, match="duplicate synthesized id"):
        build_rows(
            [{"id": "x", "sentence": "文"}],
            [first, second],
            min_duration=0.05,
            max_duration=2.0,
        )

    rows, summary = build_rows(
        [{"id": "missing", "sentence": "文"}],
        [first],
        min_duration=0.05,
        max_duration=2.0,
    )
    assert rows == []
    assert summary["missing_ids"] == ["missing"]
    assert summary["extra_synthesized_ids"] == ["x"]


def test_build_rows_accepts_an_empty_shard_manifest(tmp_path: Path):
    populated = tmp_path / "populated"
    empty = tmp_path / "empty"
    write_wav(populated / "wav" / "x.wav")
    write_manifest(populated, [{"id": "x", "wav": "wav/x.wav", "duration_sec": 0.2}])
    write_manifest(empty, [])

    rows, _summary = build_rows(
        [{"id": "x", "sentence": "文"}],
        [empty, populated],
        min_duration=0.05,
        max_duration=2.0,
    )

    assert [row["id"] for row in rows] == ["x"]
