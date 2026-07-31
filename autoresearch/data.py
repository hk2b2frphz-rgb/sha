"""Deterministic dev / holdout splitting for the search loop.

The loop optimizes against the dev split only. The holdout split is decoded
exactly once, for the winning config, so the reported final number is not the
one the search was allowed to chase.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _bucket(record_id: str, seed: int) -> float:
    digest = hashlib.md5(f"{seed}:{record_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if raw.strip():
            rows.append(json.loads(raw))
    if not rows:
        raise SystemExit(f"manifest is empty: {path}")
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def split_manifest(
    source: Path,
    dev_path: Path,
    holdout_path: Path,
    *,
    dev_ratio: float = 0.5,
    seed: int = 42,
) -> tuple[int, int]:
    """Split by a hash of the record id, so re-runs and resumes agree."""
    rows = read_jsonl(source)
    dev, holdout = [], []
    for i, row in enumerate(rows):
        record_id = str(row.get("id", i))
        (dev if _bucket(record_id, seed) < dev_ratio else holdout).append(row)
    # Never hand the loop an empty dev set (tiny manifests, unlucky hashes).
    if not dev:
        dev, holdout = holdout[: max(1, len(holdout) // 2)], holdout[max(1, len(holdout) // 2) :]
    write_jsonl(dev_path, dev)
    write_jsonl(holdout_path, holdout)
    return len(dev), len(holdout)


def collect_terms(*manifests: Path, limit: int = 60) -> list[str]:
    """Domain terms mentioned in manifests, for term recall and Whisper prompts.

    Manifests built by build_whisper_manifest.py carry no `term` field, so the
    caller normally also passes the TTS input JSONL (which does). Missing or
    unreadable files are skipped rather than failing the run.
    """
    terms: list[str] = []
    seen: set[str] = set()
    for manifest in manifests:
        if manifest is None or not Path(manifest).exists():
            continue
        try:
            rows = read_jsonl(Path(manifest))
        except (SystemExit, json.JSONDecodeError, OSError):
            continue
        for row in rows:
            raw = row.get("terms") or ([row["term"]] if row.get("term") else [])
            for term in raw:
                text = str(term).strip()
                if text and text not in seen:
                    seen.add(text)
                    terms.append(text)
            if len(terms) >= limit:
                return terms[:limit]
    return terms[:limit]
