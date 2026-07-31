import json
from pathlib import Path

from autoresearch import loop as loop_module
from autoresearch.history import STATUS_FAILED, History, Trial
from autoresearch.loop import LoopSettings, run_loop


def write_manifests(tmp_path: Path) -> tuple[Path, Path]:
    train = tmp_path / "train.jsonl"
    train.write_text(
        "".join(
            json.dumps({"id": f"{i}", "audio": f"/a/{i}.wav", "text": "文", "term": "用語"},
                       ensure_ascii=False) + "\n"
            for i in range(10)
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
    )
    defaults.update(overrides)
    return LoopSettings(**defaults)


def failing_trial(index, config, source, parent, ctx, *, budget_sec, **kwargs):  # noqa: ANN001
    return Trial(
        index=index,
        config=config,
        source=source,
        parent=parent,
        status=STATUS_FAILED,
        error="CUDA out of memory",
        seconds=1.0,
    )


def test_loop_aborts_after_consecutive_failures(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(loop_module, "run_trial", failing_trial)
    settings = make_settings(tmp_path, max_consecutive_failures=3)

    run_loop(settings)

    history = History.load_or_create(settings.work_dir / "state")
    assert len(history.trials) == 3
    assert history.best() is None
    report = (settings.work_dir / "report.md").read_text(encoding="utf-8")
    assert "failed in a row" in report


def test_isolated_failures_do_not_stop_the_search(tmp_path: Path, monkeypatch):
    calls = {"n": 0}

    def flaky(index, config, source, parent, ctx, *, budget_sec, **kwargs):  # noqa: ANN001
        calls["n"] += 1
        if index % 2 == 0:
            return failing_trial(index, config, source, parent, ctx, budget_sec=budget_sec)
        return Trial(index=index, config=config, source=source, score=0.3, seconds=1.0)

    monkeypatch.setattr(loop_module, "run_trial", flaky)
    settings = make_settings(tmp_path, max_consecutive_failures=2, max_trials=8)

    run_loop(settings)

    history = History.load_or_create(settings.work_dir / "state")
    assert len(history.trials) == 8  # ran to max_trials despite every other trial failing
    assert len(history.completed()) == 4
    assert history.best() is not None


def test_failed_trials_are_reported_and_kept_in_history(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(loop_module, "run_trial", failing_trial)
    settings = make_settings(tmp_path, max_consecutive_failures=2)

    run_loop(settings)

    trials = [json.loads(line) for line in
              (settings.work_dir / "state" / "trials.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(t["status"] == STATUS_FAILED for t in trials)
    assert all("CUDA out of memory" in t["error"] for t in trials)
    # The error text is what the proposer gets to see on the next round.
    assert not (settings.work_dir / "best_config.json").exists()


def test_max_consecutive_failures_zero_disables_the_circuit_breaker(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(loop_module, "run_trial", failing_trial)
    settings = make_settings(tmp_path, max_consecutive_failures=0, max_trials=6)

    run_loop(settings)

    history = History.load_or_create(settings.work_dir / "state")
    assert len(history.trials) == 6
