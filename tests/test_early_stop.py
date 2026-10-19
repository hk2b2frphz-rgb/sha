import json
from pathlib import Path

import pytest

from autoresearch.history import History, Trial
from autoresearch.runner import (
    TrialContext,
    build_train_command,
    early_stop_threshold,
    read_train_progress,
)
from autoresearch.space import DEFAULT_CONFIG, clamp_config
from scripts.train_whisper_lora import EarlyStopPolicy


def make_ctx(tmp_path: Path, **overrides) -> TrialContext:
    defaults = dict(
        repo_root=tmp_path,
        work_dir=tmp_path / "work",
        train_manifest=tmp_path / "train.jsonl",
        dev_manifest=tmp_path / "dev.jsonl",
        base_model_file=tmp_path / "manifest.txt",
    )
    defaults.update(overrides)
    return TrialContext(**defaults)


# -- the decision rule --------------------------------------------------------
def test_policy_stops_once_the_threshold_is_crossed():
    policy = EarlyStopPolicy(threshold=0.40, patience=0)
    should_stop, reason = policy.update(0.55)
    assert should_stop
    assert "0.5500" in reason and "0.4000" in reason


def test_policy_keeps_going_while_below_the_threshold():
    policy = EarlyStopPolicy(threshold=0.40, patience=0)
    for cer in (0.39, 0.30, 0.25):
        should_stop, reason = policy.update(cer)
        assert not should_stop and reason == ""
    assert policy.best == 0.25
    assert policy.history == [0.39, 0.30, 0.25]


def test_policy_stops_when_improvement_stalls():
    policy = EarlyStopPolicy(threshold=0.0, patience=2)
    assert policy.update(0.30) == (False, "")
    assert policy.update(0.31) == (False, "")  # 1 evaluation without improvement
    should_stop, reason = policy.update(0.32)  # 2 in a row -> give up
    assert should_stop and "not improved" in reason


def test_improvement_resets_the_patience_counter():
    policy = EarlyStopPolicy(threshold=0.0, patience=2)
    policy.update(0.30)
    policy.update(0.31)
    policy.update(0.20)  # improved again
    assert policy.since_improvement == 0
    assert policy.update(0.21) == (False, "")


def test_policy_disabled_never_stops():
    policy = EarlyStopPolicy(threshold=0.0, patience=0)
    for cer in (0.9, 0.95, 0.99, 1.0):
        assert policy.update(cer) == (False, "")


# -- the threshold the loop derives from its history --------------------------
def test_no_threshold_before_a_first_successful_trial(tmp_path: Path):
    history = History.load_or_create(tmp_path / "state")
    assert early_stop_threshold(history, make_ctx(tmp_path)) == 0.0


def test_threshold_follows_the_best_score(tmp_path: Path):
    history = History.load_or_create(tmp_path / "state")
    history.append(Trial(index=0, config=clamp_config(DEFAULT_CONFIG), score=0.40, seconds=1.0))
    history.append(Trial(index=1, config=clamp_config(DEFAULT_CONFIG), score=0.20, seconds=1.0))
    ctx = make_ctx(tmp_path, early_stop_margin=0.5)
    assert early_stop_threshold(history, ctx) == pytest.approx(0.30)  # best 0.20 * 1.5


def test_margin_zero_disables_the_threshold(tmp_path: Path):
    history = History.load_or_create(tmp_path / "state")
    history.append(Trial(index=0, config=clamp_config(DEFAULT_CONFIG), score=0.4, seconds=1.0))
    assert early_stop_threshold(history, make_ctx(tmp_path, early_stop_margin=0.0)) == 0.0


# -- wiring -------------------------------------------------------------------
def test_train_command_passes_the_early_stop_settings(tmp_path: Path):
    ctx = make_ctx(tmp_path, early_stop_patience=3, early_stop_dev_limit=16)
    cmd = build_train_command(clamp_config(DEFAULT_CONFIG), ctx, tmp_path / "t", early_stop_cer=0.42)
    assert cmd[cmd.index("--dev-manifest") + 1] == str(ctx.dev_manifest)
    assert cmd[cmd.index("--dev-limit") + 1] == "16"
    assert cmd[cmd.index("--early-stop-cer") + 1] == "0.42"
    assert cmd[cmd.index("--early-stop-patience") + 1] == "3"


def test_early_stop_can_be_switched_off_entirely(tmp_path: Path):
    ctx = make_ctx(tmp_path, early_stop_margin=0.0, early_stop_patience=0)
    cmd = build_train_command(clamp_config(DEFAULT_CONFIG), ctx, tmp_path / "t")
    assert "--dev-manifest" not in cmd
    assert "--early-stop-cer" not in cmd


def test_progress_file_is_folded_into_the_trial_metrics(tmp_path: Path):
    (tmp_path / "train_progress.json").write_text(
        json.dumps(
            {
                "dev_cer_curve": [0.6, 0.55],
                "best_dev_cer": 0.55,
                "early_stopped": True,
                "early_stop_reason": "dev CER 0.5500 exceeds the early-stop threshold 0.4000",
            }
        ),
        encoding="utf-8",
    )
    metrics = read_train_progress(tmp_path)
    assert metrics["early_stopped"] is True
    assert metrics["dev_cer_curve"] == [0.6, 0.55]
    assert "exceeds" in metrics["early_stop_reason"]


def test_missing_or_broken_progress_file_is_ignored(tmp_path: Path):
    assert read_train_progress(tmp_path) == {}
    (tmp_path / "train_progress.json").write_text("{not json", encoding="utf-8")
    assert read_train_progress(tmp_path) == {}
