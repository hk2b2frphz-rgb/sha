"""Config proposers: baseline -> random exploration -> Gemma (or evolution).

Modelled on AIDE (WecoAI), which picks the next candidate by showing an LLM the
history of previous attempts and their scores, and on OpenEvolve, which keeps a
pool of elites and mutates/recombines them.

**No silent fallback.** When the Gemma proposer is enabled and cannot produce a
usable configuration, the run stops with the reason printed. A loop that quietly
degrades to random search looks exactly like a loop that is working, which makes
it impossible to tell whether the LLM is contributing anything. Evolutionary
search is a deliberate mode (`--no-gemma`), never an automatic consolation.
"""
from __future__ import annotations

import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .history import History
from .space import (
    DEFAULT_CONFIG,
    clamp_config,
    clamp_config_with_changes,
    config_key,
    crossover,
    describe_space,
    mutate_config,
    sample_config,
)


class ProposerError(RuntimeError):
    """The proposer could not produce a usable configuration."""


@dataclass
class Proposal:
    config: dict[str, Any]
    source: str
    parent: int | None = None


class GemmaProposer:
    """Ask a local Gemma (gemma_runtime uv env) for the next configs.

    Runs as a subprocess because gemma_runtime is a separate environment from
    the training env.
    """

    def __init__(
        self,
        script: Path,
        out_dir: Path,
        model: str,
        *,
        project: str = "gemma_runtime",
        timeout_sec: int = 900,
        max_history: int = 24,
        n_proposals: int = 3,
        objective_note: str = "minimize dev CER",
        guidance: str = "",
    ) -> None:
        self.script = Path(script)
        self.out_dir = Path(out_dir)
        self.model = model
        self.project = project
        self.timeout_sec = timeout_sec
        self.max_history = max_history
        self.n_proposals = n_proposals
        self.objective_note = objective_note
        self.guidance = guidance
        self.last_error: str | None = None

    def build_request(self, history: History, avoid: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        trials = [
            {
                "index": t.index,
                "stage": t.stage,
                "config": t.config,
                "score": t.score,
                "status": t.status,
                "metrics": {
                    k: v for k, v in (t.metrics or {}).items() if k in ("cer", "wer", "term_recall")
                },
                "error": (t.error or "")[:200] or None,
            }
            for t in history.trials[-self.max_history :]
        ]
        best = history.best()
        return {
            "objective": self.objective_note,
            "space": describe_space(),
            "default_config": DEFAULT_CONFIG,
            "guidance": self.guidance,
            "best": {"index": best.index, "config": best.config, "score": best.score} if best else None,
            "history": trials,
            "already_tried": avoid or [],
            "n_proposals": self.n_proposals,
        }

    def propose(self, history: History, avoid: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        """Return clamped proposals. An empty list means the call did not work."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{len(history.trials):04d}_{len(list(self.out_dir.glob(f'request_{len(history.trials):04d}*')))}"
        request_path = self.out_dir / f"request_{tag}.json"
        response_path = self.out_dir / f"response_{tag}.json"
        request_path.write_text(
            json.dumps(self.build_request(history, avoid), ensure_ascii=False, indent=2),
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
        except subprocess.TimeoutExpired:
            self.last_error = f"gemma proposer timed out after {self.timeout_sec}s"
            return []
        except OSError as exc:
            self.last_error = f"gemma proposer could not be started: {exc}"
            return []
        if completed.returncode != 0:
            self.last_error = (
                f"gemma proposer exited with {completed.returncode}: "
                f"{(completed.stderr or completed.stdout or '')[-500:]}"
            )
            return []
        if not response_path.exists():
            self.last_error = f"gemma proposer wrote no response file: {response_path}"
            return []
        try:
            payload = json.loads(response_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            self.last_error = f"unreadable proposer response {response_path}: {exc}"
            return []
        raw_proposals = payload.get("proposals") or []
        if not raw_proposals:
            self.last_error = (
                f"gemma returned no parsable proposal ({payload.get('error') or 'empty list'}); "
                f"see {response_path}"
            )
            return []

        proposals: list[dict[str, Any]] = []
        for raw in raw_proposals:
            if not isinstance(raw, dict):
                continue
            config, changes = clamp_config_with_changes(raw)
            if changes:
                print(f"[proposer] corrected {len(changes)} value(s): {'; '.join(changes[:6])}", flush=True)
            proposals.append(config)
        self.last_error = None if proposals else "gemma proposals were not objects"
        return proposals


class ProposerPipeline:
    """Decides where each trial's config comes from and guarantees novelty."""

    def __init__(
        self,
        rng: random.Random,
        gemma: GemmaProposer | None = None,
        *,
        n_random: int = 4,
        elite_k: int = 4,
        max_attempts: int = 40,
        gemma_retries: int = 2,
    ) -> None:
        self.rng = rng
        self.gemma = gemma
        self.n_random = n_random
        self.elite_k = elite_k
        self.max_attempts = max_attempts
        self.gemma_retries = gemma_retries
        self._pending: list[dict[str, Any]] = []

    def _fresh(self, config: dict[str, Any], tried: set[str]) -> bool:
        return config_key(config) not in tried

    def propose(self, history: History, stage: str = "full") -> Proposal:
        tried = history.tried_keys()
        n_done = len(history.trials)

        if n_done == 0:
            return Proposal(clamp_config(DEFAULT_CONFIG), "baseline")

        if n_done <= self.n_random or not history.completed():
            for _ in range(self.max_attempts):
                config = sample_config(self.rng)
                if self._fresh(config, tried):
                    return Proposal(config, "random")
            raise ProposerError(
                f"random sampling produced only already-tried configs in {self.max_attempts} attempts"
            )

        elites = history.elites(self.elite_k, stage=stage) or history.elites(self.elite_k)
        best = elites[0] if elites else None

        if self.gemma is not None:
            return self._propose_with_gemma(history, tried, best)
        return self._propose_evolutionary(elites, best, tried)

    def _propose_with_gemma(self, history: History, tried: set[str], best) -> Proposal:  # noqa: ANN001
        while self._pending:
            config = self._pending.pop(0)
            if self._fresh(config, tried):
                return Proposal(config, "gemma", best.index if best else None)

        errors: list[str] = []
        for attempt in range(1, self.gemma_retries + 2):
            batch = self.gemma.propose(history, avoid=[])
            if not batch:
                message = self.gemma.last_error or "unknown proposer failure"
                print(f"[proposer] attempt {attempt} failed: {message}", flush=True)
                errors.append(message)
                continue
            self._pending = list(batch)
            while self._pending:
                config = self._pending.pop(0)
                if self._fresh(config, tried):
                    return Proposal(config, "gemma", best.index if best else None)
            message = f"all {len(batch)} proposals repeated an already evaluated config"
            print(f"[proposer] attempt {attempt} failed: {message}", flush=True)
            errors.append(message)

        raise ProposerError(
            "the Gemma proposer failed "
            f"{self.gemma_retries + 1} time(s) in a row and no fallback is used:\n  - "
            + "\n  - ".join(errors)
            + f"\nInspect {self.gemma.out_dir}, or rerun with --no-gemma for evolutionary search."
        )

    def _propose_evolutionary(self, elites: list, best, tried: set[str]) -> Proposal:  # noqa: ANN001
        for attempt in range(self.max_attempts):
            if best is None:
                config = sample_config(self.rng)
                source, parent = "random", None
            elif len(elites) >= 2 and self.rng.random() < 0.3:
                mother, father = self.rng.sample(elites, 2)
                config = crossover(mother.config, father.config, self.rng)
                source, parent = "crossover", mother.index
            else:
                seed = self.rng.choice(elites)
                n_changes = 1 + attempt // 8  # widen the step if everything collides
                config = mutate_config(seed.config, self.rng, n_changes=n_changes)
                source, parent = "mutation", seed.index
            if self._fresh(config, tried):
                return Proposal(config, source, parent)
        raise ProposerError(
            f"evolutionary search produced only already-tried configs in {self.max_attempts} attempts; "
            "the search space is likely exhausted"
        )
