"""Normalization utilities for Japanese reading evaluation."""
from __future__ import annotations

import re
import unicodedata

KATAKANA_START = ord("ァ")
KATAKANA_END = ord("ヶ")
HIRAGANA_START = ord("ぁ")
KANA_OFFSET = 0x60

SYMBOL_TRANSLATION = str.maketrans(
    {
        "〜": "～",
        "∼": "～",
        "～": "～",
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "―": "-",
        "−": "-",
        "ｰ": "ー",
    }
)


def normalize_unicode(text: str) -> str:
    """Apply Unicode compatibility normalization and known symbol folding."""
    return unicodedata.normalize("NFKC", text or "").translate(SYMBOL_TRANSLATION)


def kata_to_hira(text: str) -> str:
    """Convert full-width katakana to hiragana after NFKC half-width folding."""
    normalized = normalize_unicode(text)
    chars: list[str] = []
    for char in normalized:
        code = ord(char)
        if KATAKANA_START <= code <= KATAKANA_END:
            chars.append(chr(code - KANA_OFFSET))
        else:
            chars.append(char)
    return "".join(chars)


def hira_to_kata(text: str) -> str:
    """Convert hiragana to full-width katakana after Unicode normalization."""
    normalized = normalize_unicode(text)
    chars: list[str] = []
    for char in normalized:
        code = ord(char)
        if HIRAGANA_START <= code <= ord("ゖ"):
            chars.append(chr(code + KANA_OFFSET))
        else:
            chars.append(char)
    return "".join(chars)


def normalize_reading(text: str) -> str:
    """Normalize readings to hiragana while preserving the long vowel mark."""
    text = kata_to_hira(text)
    return re.sub(r"\s+", "", text.strip())

