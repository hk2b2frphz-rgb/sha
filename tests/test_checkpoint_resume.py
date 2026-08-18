import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_whisper_lora import latest_checkpoint  # noqa: E402


def make_checkpoint(out_dir: Path, step: int) -> Path:
    path = out_dir / "trainer" / f"checkpoint-{step}"
    path.mkdir(parents=True, exist_ok=True)
    (path / "trainer_state.json").write_text("{}", encoding="utf-8")
    return path


def test_returns_none_when_nothing_has_been_saved(tmp_path):
    assert latest_checkpoint(tmp_path) is None


def test_returns_none_when_trainer_dir_is_absent(tmp_path):
    (tmp_path / "adapter").mkdir()

    assert latest_checkpoint(tmp_path) is None


def test_picks_the_highest_step_not_the_lexicographic_max(tmp_path):
    make_checkpoint(tmp_path, 90)
    expected = make_checkpoint(tmp_path, 1000)

    # "checkpoint-90" sorts after "checkpoint-1000" as a string, so a naive
    # sorted()[-1] would resume from the older checkpoint.
    assert latest_checkpoint(tmp_path) == expected


def test_ignores_unparsable_and_non_directory_entries(tmp_path):
    expected = make_checkpoint(tmp_path, 5)
    (tmp_path / "trainer" / "checkpoint-latest").mkdir()
    (tmp_path / "trainer" / "checkpoint-7").write_text("not a dir", encoding="utf-8")

    assert latest_checkpoint(tmp_path) == expected


def test_accepts_a_string_out_dir(tmp_path):
    expected = make_checkpoint(tmp_path, 3)

    assert latest_checkpoint(str(tmp_path)) == expected


def test_pbs_script_wires_checkpoint_repo_and_resume():
    text = (ROOT / "scripts" / "run_whisper_train.pbs").read_text(encoding="utf-8")

    assert 'CHECKPOINT_REPO="${CHECKPOINT_REPO:-}"' in text
    assert 'RESUME="${RESUME:-0}"' in text
    assert "--checkpoint-repo" in text and "--resume" in text


def test_vast_runner_enables_checkpoint_persistence():
    text = (ROOT / "scripts" / "run_whisper_train_vast.sh").read_text(encoding="utf-8")

    # A rented box is destroyed on exit, so both must be on by default there.
    assert "CHECKPOINT_REPO" in text
    assert "RESUME" in text
