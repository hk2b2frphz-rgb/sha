"""Saved-prediction LLM resolvers."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from eval.normalize import normalize_reading

from .base import Resolution, Resolver

TURN_RE = re.compile(r"<\s*turn\|?>|turn\|>", re.IGNORECASE)


def clean_llm_output(text: str) -> str:
    return normalize_reading(TURN_RE.sub("", text or ""))


class LlmResolver(Resolver):
    """Resolver backed by saved *_neweval.json files; no live inference."""

    def __init__(self, name: str, predictions_path: Path | str | Iterable[Path | str]) -> None:
        super().__init__(name)
        if isinstance(predictions_path, (str, Path)):
            self.predictions_paths = [Path(predictions_path)]
        else:
            self.predictions_paths = [Path(path) for path in predictions_path]
        self.predictions_path = self.predictions_paths[0]
        self.predictions = self._load_all_predictions(self.predictions_paths)

    @staticmethod
    def _load_predictions(path: Path) -> dict[str, str]:
        data = json.loads(path.read_text(encoding="utf-8"))
        records = data.get("all_results") if isinstance(data, dict) else None
        if not isinstance(records, list):
            records = []
            for item in data.get("per_term", []):
                examples = item.get("examples") or []
                output = examples[0].get("llm_output", "") if examples else ""
                records.append({"term": item.get("term", ""), "llm_output": output})
        predictions: dict[str, str] = {}
        for record in records:
            term = str(record.get("term", ""))
            if term:
                predictions[term] = clean_llm_output(str(record.get("llm_output", "")))
        return predictions

    @classmethod
    def _load_all_predictions(cls, paths: list[Path]) -> dict[str, str]:
        predictions: dict[str, str] = {}
        for path in paths:
            predictions.update(cls._load_predictions(path))
        return predictions

    def resolve_details(self, term: str) -> Resolution:
        reading = self.predictions.get(term, "")
        return Resolution(
            term=term,
            reading=reading,
            resolver=self.name,
            confidence="saved_prediction" if reading else "missing",
            source=";".join(str(path) for path in self.predictions_paths),
            metadata={"found": bool(reading)},
        )


def build_llm_resolvers(experiments_dir: Path | str = "experiments") -> list[LlmResolver]:
    directory = Path(experiments_dir)
    grouped: dict[str, list[Path]] = {}
    for path in sorted(directory.glob("*_neweval.json")):
        lower = path.name.lower()
        if "e2b" in lower:
            name = "gemma_e2b_saved"
        elif "e4b" in lower:
            name = "gemma_e4b_saved"
        else:
            name = f"llm_saved_{path.stem}"
        grouped.setdefault(name, []).append(path)
    return [LlmResolver(name, paths) for name, paths in grouped.items()]
