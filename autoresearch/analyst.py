"""Let the local Gemma write the discussion section of the run report.

The mechanical tables in report.py say *what* happened; this says what it looks
like it means and what to try next. It runs once, after the search has stopped,
inside the reserved time at the end of the job. Any failure is non-fatal: the
report is simply written without the discussion.
"""
from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from .history import History
from .space import SPACE


def summarize_knobs(history: History, top_k: int = 5) -> dict[str, Any]:
    """Value distribution among the best trials vs. everything that was tried."""
    elites = history.elites(top_k)
    completed = history.completed()
    summary: dict[str, Any] = {}
    for name in SPACE:
        top = Counter(str(t.config.get(name)) for t in elites)
        overall = Counter(str(t.config.get(name)) for t in completed)
        summary[name] = {
            "in_top_trials": dict(top.most_common(3)),
            "in_all_trials": dict(overall.most_common(5)),
        }
    return summary


class GemmaAnalyst:
    def __init__(
        self,
        script: Path,
        out_dir: Path,
        model: str,
        *,
        project: str = "gemma_runtime",
        timeout_sec: int = 900,
        top_k: int = 8,
        enabled: bool = True,
    ) -> None:
        self.script = Path(script)
        self.out_dir = Path(out_dir)
        self.model = model
        self.project = project
        self.timeout_sec = timeout_sec
        self.top_k = top_k
        self.enabled = enabled
        self.last_error: str | None = None

    def build_request(self, history: History, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        best = history.best()
        baseline = next((t for t in history.trials if t.source == "baseline" and t.ok), None)
        failures = [t for t in history.trials if not t.ok]
        return {
            "objective": history.run_meta.get("objective", "cer"),
            "n_trials": len(history.trials),
            "n_completed": len(history.completed()),
            "gpu_hours": round(history.spent_seconds / 3600.0, 2),
            "baseline": {"index": baseline.index, "score": baseline.score} if baseline else None,
            "best": {"index": best.index, "score": best.score, "config": best.config} if best else None,
            "leaderboard": [
                {
                    "index": t.index,
                    "score": t.score,
                    "source": t.source,
                    "minutes": round(t.seconds / 60.0, 1),
                    "early_stopped": bool((t.metrics or {}).get("early_stopped")),
                    "config": t.config,
                    "metrics": {
                        k: v
                        for k, v in (t.metrics or {}).items()
                        if k in ("cer", "wer", "term_recall")
                    },
                }
                for t in history.elites(self.top_k)
            ],
            "worst": [
                {"index": t.index, "score": t.score, "config": t.config}
                for t in sorted(history.completed(), key=lambda t: -t.score)[:3]
            ],
            "failures": [
                {"index": t.index, "status": t.status, "error": (t.error or "")[:200]}
                for t in failures[:8]
            ],
            "knobs": summarize_knobs(history),
            "context": extra or {},
        }

    def analyze(self, history: History, extra: dict[str, Any] | None = None) -> str | None:
        if not self.enabled or not history.completed():
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        request_path = self.out_dir / "analysis_request.json"
        response_path = self.out_dir / "analysis_response.json"
        request_path.write_text(
            json.dumps(self.build_request(history, extra), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        cmd = [
            "uv", "run", "--project", self.project, "python", str(self.script),
            "--request", str(request_path),
            "--out", str(response_path),
            "--model", self.model,
        ]
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                encoding="utf-8",
                errors="replace",
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            self.last_error = f"analyst failed to run: {exc}"
            return None
        if completed.returncode != 0 or not response_path.exists():
            self.last_error = (completed.stderr or completed.stdout or "")[-500:]
            return None
        try:
            payload = json.loads(response_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            self.last_error = f"unreadable analyst response: {exc}"
            return None
        text = str(payload.get("analysis") or "").strip()
        self.last_error = None if text else "analyst returned an empty analysis"
        return text or None
