import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rebase_manifest_paths import index_wavs, rebase_rows  # noqa: E402


def make_wavs(root: Path, names: list[str]) -> None:
    (root / "wav").mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / "wav" / name).write_bytes(b"RIFF")


def test_rebases_audio_onto_the_new_root():
    rows = [{"id": "gs_0001", "audio": "/old/box/wav/gs_0001.wav", "text": "冠動脈"}]
    root = Path("/tmp")  # not used; index passed directly
    index = {"gs_0001.wav": Path("/workspace/data/wav/gs_0001.wav")}
    out, missing = rebase_rows(rows, index, allow_missing=False)

    assert missing == []
    assert out[0]["audio"] == "/workspace/data/wav/gs_0001.wav"
    assert out[0]["text"] == "冠動脈", "unrelated fields must survive"
    assert root.exists()


def test_missing_wav_fails_fast_by_default():
    rows = [{"id": "gs_0001", "audio": "/old/wav/gs_0001.wav"}]
    with pytest.raises(SystemExit):
        rebase_rows(rows, {}, allow_missing=False)


def test_missing_wav_is_dropped_when_allowed():
    rows = [
        {"id": "gs_0001", "audio": "/old/wav/gs_0001.wav"},
        {"id": "gs_0002", "audio": "/old/wav/gs_0002.wav"},
    ]
    index = {"gs_0002.wav": Path("/new/wav/gs_0002.wav")}
    out, missing = rebase_rows(rows, index, allow_missing=True)

    assert [r["id"] for r in out] == ["gs_0002"]
    assert missing == ["gs_0001.wav"]


def test_index_finds_wavs_in_nested_shard_dirs(tmp_path):
    make_wavs(tmp_path / "shard_00", ["gs_0001.wav"])
    make_wavs(tmp_path / "shard_01", ["gs_0002.wav"])

    index = index_wavs(tmp_path)

    assert set(index) == {"gs_0001.wav", "gs_0002.wav"}


def test_duplicate_basenames_are_rejected(tmp_path):
    make_wavs(tmp_path / "shard_00", ["gs_0001.wav"])
    make_wavs(tmp_path / "shard_01", ["gs_0001.wav"])

    with pytest.raises(SystemExit):
        index_wavs(tmp_path)


def test_round_trip_through_a_real_directory(tmp_path):
    make_wavs(tmp_path, ["gs_0001.wav"])
    rows = [{"id": "gs_0001", "audio": "/somewhere/else/gs_0001.wav"}]

    out, _ = rebase_rows(rows, index_wavs(tmp_path), allow_missing=False)

    assert Path(out[0]["audio"]).exists()
    assert json.loads(json.dumps(out[0]))["id"] == "gs_0001"
