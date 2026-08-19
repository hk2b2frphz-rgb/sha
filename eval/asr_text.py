"""Shared text normalization and CER/WER helpers for ASR evaluation.

`evaluate_whisper_streaming.py` (whisper-streaming) and `eval_whisper_hf.py`
(fast batched HF decode used by the auto-research loop) must score identically,
so both import the metric code from here.
"""
from __future__ import annotations

import re
import unicodedata

from .metrics import levenshtein

JA_PUNCT_RE = re.compile(r"[\s、。．，,.!?！？「」『』（）()［］\[\]【】:：;；・…~〜\-ー_\"'“”‘’]+")


def normalize_text(text: str, *, keep_spaces: bool = False) -> str:
    """NFKC + lowercase + punctuation stripping used for both CER and WER."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = JA_PUNCT_RE.sub(" " if keep_spaces else "", text)
    return " ".join(text.split()) if keep_spaces else text.strip()


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    return levenshtein(ref, hyp)


class JapaneseTokenizer:
    """Word tokenizer for WER; falls back to characters when fugashi is absent."""

    def __init__(self, enabled: bool) -> None:
        self.mode = "char"
        self.tagger = None
        if enabled:
            try:
                import fugashi

                self.tagger = fugashi.Tagger()
                self.mode = "fugashi"
            except Exception:
                self.tagger = None

    def words(self, text: str) -> list[str]:
        text = normalize_text(text, keep_spaces=True)
        if not text:
            return []
        if self.tagger is not None:
            return [word.surface for word in self.tagger(text) if word.surface.strip()]
        if " " in text:
            return text.split()
        return list(normalize_text(text))


def error_rate(ref_units: list[str], hyp_units: list[str]) -> float:
    if not ref_units:
        return 0.0 if not hyp_units else 1.0
    return edit_distance(ref_units, hyp_units) / len(ref_units)


def term_hit(term: str, hypothesis: str) -> bool:
    """True when a domain term survives ASR (normalized substring match)."""
    normalized_term = normalize_text(term)
    return bool(normalized_term) and normalized_term in normalize_text(hypothesis)
