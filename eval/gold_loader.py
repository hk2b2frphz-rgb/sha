"""Load legacy and extended gold reading TSV files."""
from __future__ import annotations

import csv
from pathlib import Path

from .gold_schema import GoldEntry, GoldReading
from .normalize import normalize_reading


def _merge(entries: dict[str, GoldEntry], term: str, reading: str, source: str, domain: str = "") -> None:
    normalized = normalize_reading(reading)
    if not term or not normalized:
        return
    existing = entries.get(term)
    if existing is None:
        entries[term] = GoldEntry(term=term, readings=(GoldReading(normalized, source),), domain=domain)
        return
    readings = list(existing.readings)
    if normalized not in {r.reading for r in readings}:
        readings.append(GoldReading(normalized, source))
    entries[term] = GoldEntry(term=term, readings=tuple(readings), domain=existing.domain or domain)


def load_gold_tsv(path: Path | str) -> dict[str, GoldEntry]:
    """Load term readings from legacy TSV or extended TSV.

    The gold side is intentionally only a loader for externally curated
    references. Do not generate these readings from Sudachi, UniDic,
    pyopenjtalk, or any future method under evaluation; doing so would create
    circular evaluation and inflate scores for dictionary-backed systems.
    """
    path = Path(path)
    entries: dict[str, GoldEntry] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header: list[str] | None = None
        for row in reader:
            if not row or not row[0] or row[0].startswith("#"):
                continue
            if header is None and any(cell in {"term", "reading", "readings"} for cell in row):
                header = row
                continue
            if header:
                values = {header[i]: row[i] for i in range(min(len(header), len(row)))}
                term = values.get("term", "")
                readings_field = values.get("readings") or values.get("reading") or ""
                source = values.get("source", "legacy_gold")
                domain = values.get("domain", "")
            else:
                term = row[1] if len(row) >= 4 and row[0].isdigit() else row[0]
                readings_field = row[2] if len(row) >= 4 and row[0].isdigit() else (row[1] if len(row) > 1 else "")
                source = "legacy_gold"
                domain = row[4] if len(row) >= 5 else ""
            for reading in readings_field.replace(";", "|").split("|"):
                _merge(entries, term.strip(), reading.strip(), source, domain.strip())
    return entries


def load_gold_files(paths: list[Path | str]) -> dict[str, GoldEntry]:
    merged: dict[str, GoldEntry] = {}
    for path in paths:
        for term, entry in load_gold_tsv(path).items():
            for reading in entry.readings:
                _merge(merged, term, reading.reading, reading.source, entry.domain)
    return merged

