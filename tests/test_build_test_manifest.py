import json
import subprocess
import sys
from pathlib import Path


def test_plain_text_references_follow_natural_wav_order(tmp_path: Path):
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    for name in ("10.wav", "2.wav", "1.wav"):
        (wav_dir / name).touch()
    refs = tmp_path / "refs.txt"
    refs.write_text("first\nsecond\ntenth\n", encoding="utf-8")
    output = tmp_path / "manifest.jsonl"

    subprocess.run(
        [
            sys.executable,
            "scripts/build_test_manifest.py",
            "--wav-dir",
            str(wav_dir),
            "--references",
            str(refs),
            "--out",
            str(output),
        ],
        check=True,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == ["1", "2", "10"]
    assert [row["sentence"] for row in rows] == ["first", "second", "tenth"]
