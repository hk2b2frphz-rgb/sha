"""Common resolver interface for Japanese reading estimation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from eval.normalize import normalize_reading


@dataclass(frozen=True)
class Resolution:
    """A resolver output plus metadata used by hybrids and reports."""

    term: str
    reading: str
    resolver: str
    confidence: str = "unknown"
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def normalized_reading(self) -> str:
        return normalize_reading(self.reading)


class Resolver(ABC):
    """Abstract base class: term string in, hiragana reading string out."""

    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def resolve_details(self, term: str) -> Resolution:
        """Return the reading with method-specific metadata."""

    def resolve(self, term: str) -> str:
        """Return only the normalized hiragana reading for evaluation."""
        return self.resolve_details(term).normalized_reading
