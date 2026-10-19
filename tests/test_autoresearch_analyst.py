import json
from pathlib import Path

from autoresearch.analyst import GemmaAnalyst, summarize_knobs
from autoresearch.history import STATUS_FAILED, History, Trial
from autoresearch.report import write_report
from autoresearch.space import DEFAULT_CONFIG, clamp_config


def build_history(tmp_path: Path) -> History:
    history = History.load_or_create(tmp_path / "state", run_meta={"objective": "cer"})
    history.append(
        Trial(
            index=0,
            config=clamp_config(DEFAULT_CONFIG),
            source="baseline",
            score=0.40,
            metrics={"cer": 0.40, "wer": 0.55, "term_recall": 0.6},
            seconds=1800.0,
        )
    )
    history.append(
        Trial(
            index=1,
            config=clamp_config({**DEFAULT_CONFIG, "target_modules": "qkvo_fc", "lora_r": 64}),
            source="gemma",
            score=0.31,
            metrics={"cer": 0.31, "early_stopped": False},
            seconds=2400.0,
        )
    )
    history.append(
        Trial(
            index=2,
            config=clamp_config({**DEFAULT_CONFIG, "batch_size": 16, "lora_r": 128}),
            source="random",
            status=STATUS_FAILED,
            error="CUDA out of memory",
            seconds=300.0,
        )
    )
    return history


def test_request_carries_baseline_best_failures_and_knobs(tmp_path: Path):
    history = build_history(tmp_path)
    analyst = GemmaAnalyst(script=Path("x"), out_dir=tmp_path, model="stub")

    request = analyst.build_request(history, {"stop_reason": "budget exhausted"})

    assert request["n_trials"] == 3 and request["n_completed"] == 2
    assert request["baseline"]["index"] == 0
    assert request["best"]["index"] == 1
    assert [row["index"] for row in request["leaderboard"]] == [1, 0]
    assert request["failures"][0]["error"].startswith("CUDA out of memory")
    assert request["context"]["stop_reason"] == "budget exhausted"
    assert request["knobs"]["target_modules"]["in_top_trials"]["qkvo_fc"] == 1


def test_summarize_knobs_counts_top_and_overall(tmp_path: Path):
    knobs = summarize_knobs(build_history(tmp_path), top_k=1)
    # Only trial 1 is in the top; both completed trials count in the overall view.
    assert knobs["lora_r"]["in_top_trials"] == {"64": 1}
    assert knobs["lora_r"]["in_all_trials"] == {"64": 1, "32": 1}


def test_analysis_is_skipped_when_disabled_or_empty(tmp_path: Path):
    history = build_history(tmp_path)
    disabled = GemmaAnalyst(script=Path("x"), out_dir=tmp_path, model="stub", enabled=False)
    assert disabled.analyze(history) is None

    empty_history = History.load_or_create(tmp_path / "empty")
    enabled = GemmaAnalyst(script=Path("x"), out_dir=tmp_path, model="stub")
    assert enabled.analyze(empty_history) is None


def test_analyst_failure_is_reported_but_not_raised(tmp_path: Path):
    history = build_history(tmp_path)
    # The script does not exist, so the subprocess call fails.
    analyst = GemmaAnalyst(script=tmp_path / "missing.py", out_dir=tmp_path, model="stub")
    assert analyst.analyze(history) is None
    assert analyst.last_error


def test_report_marks_the_analysis_as_machine_written(tmp_path: Path):
    history = build_history(tmp_path)
    out = tmp_path / "report.md"

    write_report(history, out, {"stop_reason": "done"}, "### 効いた設定\n- qkvo_fc が有利", "gemma-x")
    text = out.read_text(encoding="utf-8")
    assert "## 考察 — 自動生成 (gemma-x)" in text
    assert "検証されていない" in text
    assert "qkvo_fc が有利" in text

    write_report(history, out, {"stop_reason": "done"})
    assert "考察" not in out.read_text(encoding="utf-8")


def test_request_is_json_serializable(tmp_path: Path):
    request = GemmaAnalyst(script=Path("x"), out_dir=tmp_path, model="stub").build_request(
        build_history(tmp_path)
    )
    assert json.loads(json.dumps(request, ensure_ascii=False))["n_trials"] == 3
