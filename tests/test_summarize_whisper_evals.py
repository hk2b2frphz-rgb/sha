import json
import subprocess
import sys
from pathlib import Path


def test_summary_ranks_by_cer_then_wer_then_speed(tmp_path: Path):
    models = tmp_path / "models.tsv"
    models.write_text("fast\t/model/fast\naccurate\t/model/accurate\n", encoding="utf-8")
    for name, cer, wer, speed in (("fast", 0.1, 0.2, 4.0), ("accurate", 0.1, 0.1, 1.0)):
        output = tmp_path / "results" / name
        output.mkdir(parents=True)
        (output / "summary.json").write_text(
            json.dumps({"cer": cer, "wer": wer, "speed_x": speed, "rtf": 1 / speed, "n": 3, "model_dir": f"/model/{name}"}),
            encoding="utf-8",
        )

    subprocess.run(
        [sys.executable, "scripts/summarize_whisper_evals.py", "--models-file", str(models), "--results-dir", str(tmp_path / "results")],
        check=True,
    )

    report = (tmp_path / "results" / "summary.md").read_text(encoding="utf-8")
    assert "| 1 | accurate |" in report
    assert "| 2 | fast |" in report


def test_summary_includes_missing_model_as_skipped(tmp_path: Path):
    models = tmp_path / "models.tsv"
    models.write_text("missing\t/model/missing\n", encoding="utf-8")
    result = tmp_path / "results" / "missing"
    result.mkdir(parents=True)
    (result / "status.txt").write_text("status=skipped\nreason=model.bin is missing\n", encoding="utf-8")

    subprocess.run(
        [sys.executable, "scripts/summarize_whisper_evals.py", "--models-file", str(models), "--results-dir", str(tmp_path / "results")],
        check=True,
    )

    report = (tmp_path / "results" / "summary.md").read_text(encoding="utf-8")
    assert "| - | missing | skipped |" in report
    assert "model.bin is missing" in report
