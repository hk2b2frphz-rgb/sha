"""Heuristic error type labels for reading evaluation."""
from __future__ import annotations

from .metrics import ReadingScore
from .normalize import normalize_reading


def classify_error(raw_prediction: str, references: list[str], score: ReadingScore) -> str:
    """Classify result using deterministic, inspectable heuristics.

    Labels are meant for analysis rather than adjudication:
    - exact/correct alternative readings are accepted as "別解".
    - "正規化差異" isolates legacy failures caused only by kana width/script.
    - MER <= 0.5 is a partial misreading; larger distance is a complete one.
    """
    normalized_prediction = normalize_reading(raw_prediction)
    normalized_refs = [normalize_reading(ref) for ref in references]

    if score.exact_match:
        if len(set(normalized_refs)) > 1 and normalized_prediction != normalized_refs[0]:
            return "別解"
        return "正解"
    if raw_prediction.strip() in references or normalized_prediction in normalized_refs:
        return "正規化差異"
    if score.mora_error_rate <= 0.5:
        return "部分誤読"
    return "完全誤読"

