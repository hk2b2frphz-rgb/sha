import json
from pathlib import Path

from autoresearch.history import STATUS_FAILED, History, Trial
from autoresearch.space import DEFAULT_CONFIG, clamp_config


def make_trial(index: int, score: float | None, status: str = "completed") -> Trial:
    config = clamp_config({**DEFAULT_CONFIG, "lora_r": [8, 16, 32, 64, 128][index % 5]})
    return Trial(index=index, config=config, score=score, status=status, seconds=60.0 * (index + 1))


def test_history_round_trips_through_disk(tmp_path: Path):
    history = History.load_or_create(tmp_path)
    history.append(make_trial(0, 0.30))
    history.append(make_trial(1, 0.25))
    history.append(make_trial(2, None, STATUS_FAILED))

    reloaded = History.load_or_create(tmp_path)
    assert len(reloaded.trials) == 3
    assert reloaded.next_index() == 3
    assert reloaded.best().index == 1
    assert [t.index for t in reloaded.elites(2)] == [1, 0]
    assert reloaded.spent_seconds == history.spent_seconds

    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["best_index"] == 1
    assert state["n_trials"] == 3
    assert len((tmp_path / "trials.jsonl").read_text(encoding="utf-8").strip().splitlines()) == 3


def test_failed_trials_are_never_selected_as_best(tmp_path: Path):
    history = History.load_or_create(tmp_path)
    history.append(make_trial(0, None, STATUS_FAILED))
    assert history.best() is None
    assert history.completed() == []
    history.append(make_trial(1, 0.4))
    assert history.best().index == 1


def test_tried_keys_lets_the_proposer_avoid_repeats(tmp_path: Path):
    history = History.load_or_create(tmp_path)
    trial = make_trial(0, 0.3)
    history.append(trial)
    from autoresearch.space import config_key

    assert config_key(trial.config) in history.tried_keys()


def test_mean_trial_seconds_uses_the_recent_window(tmp_path: Path):
    history = History.load_or_create(tmp_path)
    for index in range(6):
        history.append(make_trial(index, 0.5 - index * 0.01))
    # last five trials: 120, 180, 240, 300, 360 seconds
    assert history.mean_trial_seconds(window=5) == 240.0
