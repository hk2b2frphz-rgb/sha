"""Gold reading schema for independent human/authority-derived references."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GoldReading:
    reading: str
    source: str = "legacy_gold"


@dataclass(frozen=True)
class GoldEntry:
    term: str
    readings: tuple[GoldReading, ...]
    domain: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def reading_values(self) -> tuple[str, ...]:
        return tuple(reading.reading for reading in self.readings)

