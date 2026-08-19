"""Reusable metrics for reading prediction evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .normalize import normalize_reading

SMALL_KANA = set("ゃゅょぁぃぅぇぉゎゕゖ")


def levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    """Compute Levenshtein edit distance over arbitrary token sequences."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if ca == cb else 1),
                )
            )
        previous = current
    return previous[-1]


def morae(reading: str) -> list[str]:
    """Split normalized kana into mora-like units for MER.

    Contracted sounds attach small kana to the preceding base kana. The long
    vowel mark is kept as its own mora because TTS output often preserves it
    explicitly and collapsing it would hide clinically relevant errors.
    """
    units: list[str] = []
    for char in normalize_reading(reading):
        if char in SMALL_KANA and units:
            units[-1] += char
        else:
            units.append(char)
    return units


def exact_match(prediction: str, reference: str) -> bool:
    return normalize_reading(prediction) == normalize_reading(reference)


def character_distance(prediction: str, reference: str) -> int:
    return levenshtein(list(normalize_reading(prediction)), list(normalize_reading(reference)))


def mora_distance(prediction: str, reference: str) -> int:
    return levenshtein(morae(prediction), morae(reference))


def mora_error_rate(prediction: str, reference: str) -> float:
    ref_morae = morae(reference)
    if not ref_morae:
        return 0.0 if not normalize_reading(prediction) else 1.0
    return mora_distance(prediction, reference) / len(ref_morae)


@dataclass(frozen=True)
class ReadingScore:
    reference: str
    exact_match: bool
    character_distance: int
    mora_distance: int
    mora_error_rate: float


def score_against_reference(prediction: str, reference: str) -> ReadingScore:
    return ReadingScore(
        reference=reference,
        exact_match=exact_match(prediction, reference),
        character_distance=character_distance(prediction, reference),
        mora_distance=mora_distance(prediction, reference),
        mora_error_rate=mora_error_rate(prediction, reference),
    )


def best_reference_score(prediction: str, references: Iterable[str]) -> ReadingScore:
    """Return the best score over multiple acceptable references."""
    refs = list(references)
    if not refs:
        raise ValueError("at least one reference reading is required")
    scores = [score_against_reference(prediction, ref) for ref in refs]
    return min(scores, key=lambda s: (not s.exact_match, s.mora_error_rate, s.character_distance))

