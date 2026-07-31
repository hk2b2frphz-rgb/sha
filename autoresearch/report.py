"""Markdown reporting for an auto-research run."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .history import History
from .space import SPACE


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return "-" if value is None else str(value)


def leaderboard_markdown(history: History, limit: int = 15) -> str:
    """Ranked trials.

    Screening and full-length trials appear in one table, but their scores come
    from different amounts of training, so the stage column is not decoration:
    a screening row must never be read as beating a full-length row.
    """
    rows = history.leaderboard(limit)
    if not rows:
        return "(no completed trials)\n"
    lines = [
        "| rank | trial | stage | score | CER | WER | term recall | source | minutes |",
        "|---:|---:|:--|---:|---:|---:|---:|:--|---:|",
    ]
    for rank, trial in enumerate(rows, 1):
        metrics = trial.metrics or {}
        early = " (early-stopped)" if metrics.get("early_stopped") else ""
        lines.append(
            f"| {rank} | {trial.index} | {trial.stage}{early} | {_fmt(trial.score)} | "
            f"{_fmt(metrics.get('cer'))} | {_fmt(metrics.get('wer'))} | "
            f"{_fmt(metrics.get('term_recall'))} | {trial.source} | {trial.seconds / 60:.1f} |"
        )
    return "\n".join(lines) + "\n"


def knob_summary(history: History, top_k: int = 5) -> str:
    """Which values show up in the best trials vs. everything tried."""
    elites = history.elites(top_k)
    if not elites:
        return "(no completed trials)\n"
    lines = ["| parameter | best trial | values among top trials |", "|:--|:--|:--|"]
    best = elites[0]
    for name in SPACE:
        values = [str(t.config.get(name)) for t in elites]
        counts: dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        ranked = ", ".join(f"{v} x{c}" for v, c in sorted(counts.items(), key=lambda kv: -kv[1]))
        lines.append(f"| {name} | {_fmt(best.config.get(name))} | {ranked} |")
    return "\n".join(lines) + "\n"


def write_report(
    history: History,
    out_path: Path,
    extra: dict[str, Any] | None = None,
    analysis: str | None = None,
    analysis_model: str | None = None,
) -> None:
    # The headline number is the one that ships: a full-length trial when the
    # run got that far, never a screening result.
    best = history.final_best()
    completed = history.completed()
    baseline = next((t for t in history.trials if t.source == "baseline" and t.ok), None)
    lines = [
        "# Whisper auto-research run",
        "",
        f"- trials: {len(history.trials)} (completed {len(completed)}, "
        f"failed {len(history.trials) - len(completed)})",
        f"- GPU time spent: {history.spent_seconds / 3600:.2f} h",
        f"- sessions (PBS jobs): {len(history.sessions)}",
    ]
    if baseline is not None:
        lines.append(f"- baseline score: {_fmt(baseline.score)} (trial {baseline.index})")
    if best is not None:
        lines.append(
            f"- best score: {_fmt(best.score)} (trial {best.index}, stage={best.stage}, {best.source})"
        )
        if baseline is not None and baseline.score:
            delta = (baseline.score - best.score) / baseline.score * 100
            lines.append(f"- improvement over baseline: {delta:.1f}%")
    for key, value in (extra or {}).items():
        lines.append(f"- {key}: {value}")
    lines += ["", "## Leaderboard", "", leaderboard_markdown(history), "", "## Knobs in the top trials", "", knob_summary(history)]
    if best is not None:
        lines += [
            "",
            "## Best configuration",
            "",
            "```json",
            json.dumps(best.config, ensure_ascii=False, indent=2, sort_keys=True),
            "```",
        ]
    if analysis:
        # Clearly fenced off: this section is model-written prose about the
        # tables above, not measured data.
        model_note = f" ({analysis_model})" if analysis_model else ""
        lines += [
            "",
            f"## 考察 — 自動生成{model_note}",
            "",
            "> 以下はローカル LLM が上の表だけを見て書いた文章で、検証されていない。",
            "> 数値の根拠は必ず上の表と `state/trials.jsonl` で確認すること。",
            "",
            analysis.strip(),
        ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
