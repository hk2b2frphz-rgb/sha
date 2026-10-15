import json
from pathlib import Path

import pytest

from autoresearch import loop as loop_module
from autoresearch.history import STAGE_FULL, STAGE_SCREEN, History, Trial
from autoresearch.loop import LoopSettings, run_loop
from autoresearch.runner import TrialContext, build_train_command, early_stop_threshold
from autoresearch.space import DEFAULT_CONFIG, clamp_config, config_key


def write_manifests(tmp_path: Path) -> tuple[Path, Path]:
    train = tmp_path / "train.jsonl"
    train.write_text(
        "".join(
            json.dumps({"id": str(i), "audio": f"/a/{i}.wav", "text": "文"}) + "\n" for i in range(10)
        ),
        encoding="utf-8",
    )
    test = tmp_path / "test.jsonl"
    test.write_text(
        "".join(
            json.dumps({"id": f"t{i}", "wav": f"/a/t{i}.wav", "sentence": "文"}) + "\n"
            for i in range(10)
        ),
        encoding="utf-8",
    )
    return train, test


def make_settings(tmp_path: Path, **overrides) -> LoopSettings:
    train, test = write_manifests(tmp_path)
    defaults = dict(
        repo_root=tmp_path,
        work_dir=tmp_path / "work",
        train_manifest=train,
        eval_manifest=test,
        mock=True,
        use_gemma=False,
        budget_hours=1.0,
        reserve_minutes=0.0,
        max_trials=6,
        screen_minutes=5.0,
        finalists=2,
    )
    defaults.update(overrides)
    return LoopSettings(**defaults)


def record_trials(monkeypatch) -> list[dict]:
    """Replace run_trial with a recorder that scores by lora_r (deterministic)."""
    seen: list[dict] = []

    def fake(index, config, source, parent, ctx, *, budget_sec, early_stop_cer=0.0,
             stage=STAGE_FULL, max_train_seconds=0.0):  # noqa: ANN001
        seen.append(
            {
                "index": index,
                "stage": stage,
                "source": source,
                "max_train_seconds": max_train_seconds,
                "config": config,
            }
        )
        # Screening scores sit higher than full ones, as short training would.
        base = 1.0 / (1 + int(config["lora_r"]))
        return Trial(
            index=index,
            config=config,
            source=source,
            parent=parent,
            stage=stage,
            score=base + (0.1 if stage == STAGE_SCREEN else 0.0),
            seconds=1.0,
        )

    monkeypatch.setattr(loop_module, "run_trial", fake)
    return seen


def test_screening_runs_first_then_the_finalists_at_full_length(tmp_path: Path, monkeypatch):
    seen = record_trials(monkeypatch)
    settings = make_settings(tmp_path)

    run_loop(settings)

    screened = [row for row in seen if row["stage"] == STAGE_SCREEN]
    full = [row for row in seen if row["stage"] == STAGE_FULL]
    assert len(screened) == settings.max_trials
    assert len(full) == settings.finalists
    # Screening caps training time; the finals do not.
    assert all(row["max_train_seconds"] == 5.0 * 60 for row in screened)
    assert all(row["max_train_seconds"] == 0.0 for row in full)
    assert all(row["source"].startswith("finalist") for row in full)
    # The finals re-run exactly the best screened configs.
    history = History.load_or_create(settings.work_dir / "state")
    best_screened = {config_key(t.config) for t in history.elites(2, stage=STAGE_SCREEN)}
    assert {config_key(row["config"]) for row in full} == best_screened


def test_the_shipped_model_comes_from_a_full_length_trial(tmp_path: Path, monkeypatch):
    record_trials(monkeypatch)
    settings = make_settings(tmp_path)

    run_loop(settings)

    history = History.load_or_create(settings.work_dir / "state")
    # Screening scores are worse here, yet best() over everything still finds a
    # full trial; what matters is that final_best never returns a screening one.
    assert history.final_best().stage == STAGE_FULL
    best_config = json.loads((settings.work_dir / "best_config.json").read_text(encoding="utf-8"))
    assert best_config == history.final_best().config


def test_screening_disabled_keeps_the_single_stage_behaviour(tmp_path: Path, monkeypatch):
    seen = record_trials(monkeypatch)
    settings = make_settings(tmp_path, screen_minutes=0.0)

    run_loop(settings)

    assert all(row["stage"] == STAGE_FULL for row in seen)
    assert all(row["max_train_seconds"] == 0.0 for row in seen)
    assert len(seen) == settings.max_trials


def test_early_stop_threshold_never_mixes_stages(tmp_path: Path):
    history = History.load_or_create(tmp_path / "state")
    history.append(
        Trial(index=0, config=clamp_config(DEFAULT_CONFIG), stage=STAGE_SCREEN, score=0.50, seconds=1.0)
    )
    history.append(
        Trial(index=1, config=clamp_config(DEFAULT_CONFIG), stage=STAGE_FULL, score=0.20, seconds=1.0)
    )
    ctx = TrialContext(
        repo_root=tmp_path,
        work_dir=tmp_path,
        train_manifest=tmp_path / "t.jsonl",
        dev_manifest=tmp_path / "d.jsonl",
        base_model_file=tmp_path / "m.txt",
        early_stop_margin=0.5,
    )
    assert early_stop_threshold(history, ctx, STAGE_SCREEN) == pytest.approx(0.75)  # 0.50 * 1.5
    assert early_stop_threshold(history, ctx, STAGE_FULL) == pytest.approx(0.30)  # 0.20 * 1.5


def test_train_command_passes_the_wall_clock_budget(tmp_path: Path):
    ctx = TrialContext(
        repo_root=tmp_path,
        work_dir=tmp_path,
        train_manifest=tmp_path / "t.jsonl",
        dev_manifest=tmp_path / "d.jsonl",
        base_model_file=tmp_path / "m.txt",
    )
    screened = build_train_command(
        clamp_config(DEFAULT_CONFIG), ctx, tmp_path / "t", max_train_seconds=1500
    )
    assert screened[screened.index("--max-train-seconds") + 1] == "1500"
    full = build_train_command(clamp_config(DEFAULT_CONFIG), ctx, tmp_path / "t")
    assert "--max-train-seconds" not in full
