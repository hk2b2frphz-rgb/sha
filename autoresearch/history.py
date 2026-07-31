"""Trial records and crash-safe run state.

State is rewritten after every trial so a PBS job that hits its walltime (or
is killed) can be resubmitted and pick up where it stopped.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .space import config_key

STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_TIMEOUT = "timeout"

# Screening trials train for a fixed short wall-clock budget so many configs can
# be compared cheaply; full trials train to completion. Scores from the two
# stages are NOT comparable, so every query that ranks trials takes a stage.
STAGE_SCREEN = "screen"
STAGE_FULL = "full"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Trial:
    index: int
    config: dict[str, Any]
    source: str = "random"
    parent: int | None = None
    stage: str = STAGE_FULL
    status: str = STATUS_COMPLETED
    score: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    started_at: str = ""
    finished_at: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == STATUS_COMPLETED and self.score is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Trial":
        known = {f: data.get(f) for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        known["index"] = int(known["index"])
        known["config"] = known["config"] or {}
        known["metrics"] = known["metrics"] or {}
        known["seconds"] = float(known["seconds"] or 0.0)
        known["source"] = known["source"] or "random"
        known["stage"] = known["stage"] or STAGE_FULL
        known["status"] = known["status"] or STATUS_COMPLETED
        return cls(**known)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class History:
    """Append-only trial log plus the derived state file."""

    def __init__(self, state_dir: Path, run_meta: dict[str, Any] | None = None) -> None:
        self.state_dir = Path(state_dir)
        self.state_path = self.state_dir / "state.json"
        self.trials_path = self.state_dir / "trials.jsonl"
        self.trials: list[Trial] = []
        self.run_meta: dict[str, Any] = dict(run_meta or {})
        self.spent_seconds: float = 0.0
        self.sessions: list[dict[str, Any]] = []

    # -- persistence ---------------------------------------------------------
    @classmethod
    def load_or_create(cls, state_dir: Path, run_meta: dict[str, Any] | None = None) -> "History":
        history = cls(state_dir, run_meta)
        if history.state_path.exists():
            data = json.loads(history.state_path.read_text(encoding="utf-8"))
            history.trials = [Trial.from_dict(t) for t in data.get("trials", [])]
            history.spent_seconds = float(data.get("spent_seconds", 0.0))
            history.sessions = list(data.get("sessions", []))
            stored_meta = data.get("run_meta") or {}
            stored_meta.update(history.run_meta)
            history.run_meta = stored_meta
        else:
            history.run_meta.setdefault("created_at", utcnow())
        return history

    def save(self) -> None:
        payload = {
            "run_meta": self.run_meta,
            "updated_at": utcnow(),
            "spent_seconds": self.spent_seconds,
            "n_trials": len(self.trials),
            "best_index": self.best().index if self.best() else None,
            "best_score": self.best().score if self.best() else None,
            "sessions": self.sessions,
            "trials": [t.to_dict() for t in self.trials],
        }
        _atomic_write(self.state_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    def append(self, trial: Trial) -> None:
        self.trials.append(trial)
        self.spent_seconds += float(trial.seconds or 0.0)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with self.trials_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(trial.to_dict(), ensure_ascii=False) + "\n")
        self.save()

    def start_session(self, info: dict[str, Any]) -> None:
        self.sessions.append({"started_at": utcnow(), **info})
        self.save()

    # -- queries -------------------------------------------------------------
    def next_index(self) -> int:
        return (max((t.index for t in self.trials), default=-1)) + 1

    def completed(self, stage: str | None = None) -> list[Trial]:
        return [t for t in self.trials if t.ok and (stage is None or t.stage == stage)]

    def best(self, stage: str | None = None) -> Trial | None:
        done = self.completed(stage)
        return min(done, key=lambda t: (t.score, t.index)) if done else None

    def elites(self, k: int, stage: str | None = None) -> list[Trial]:
        return sorted(self.completed(stage), key=lambda t: (t.score, t.index))[: max(1, k)]

    def final_best(self) -> Trial | None:
        """The trial to ship: a full-length one if the run got that far."""
        return self.best(STAGE_FULL) or self.best()

    def tried_keys(self) -> set[str]:
        return {config_key(t.config) for t in self.trials}

    def mean_trial_seconds(self, window: int = 5) -> float | None:
        durations = [t.seconds for t in self.trials if t.seconds > 0][-window:]
        return sum(durations) / len(durations) if durations else None

    def leaderboard(self, limit: int = 10, stage: str | None = None) -> list[Trial]:
        return self.elites(limit, stage)
