"""Optional dictionary-backed reading resolvers.

Dictionary tools are prediction-side methods only. The experiment runner never
uses them to create or alter gold readings.
"""
from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Callable

from eval.normalize import normalize_reading

from .base import Resolution, Resolver

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackendStatus:
    name: str
    available: bool
    message: str
    install: str


class DictResolver(Resolver):
    """A thin wrapper around an installed dictionary backend."""

    def __init__(self, name: str, backend: Callable[[str], Resolution]) -> None:
        super().__init__(name)
        self._backend = backend

    def resolve_details(self, term: str) -> Resolution:
        resolution = self._backend(term)
        return Resolution(
            term=term,
            reading=normalize_reading(resolution.reading),
            resolver=self.name,
            confidence=resolution.confidence,
            source=resolution.source or self.name,
            metadata=resolution.metadata,
        )


def _confidence(token_count: int, reading: str, unknown: bool = False) -> str:
    # A single token with a non-empty reading is the conservative high-confidence
    # case used by the first-round hybrid. Multi-token compounds may still be
    # useful, but are treated as fallback candidates rather than decisive outputs.
    if reading and token_count == 1 and not unknown:
        return "high"
    if reading:
        return "low"
    return "unavailable"


def _pyopenjtalk_backend() -> Callable[[str], Resolution]:
    pyopenjtalk = importlib.import_module("pyopenjtalk")

    def resolve(term: str) -> Resolution:
        raw = pyopenjtalk.g2p(term, kana=True)
        reading = normalize_reading(raw)
        return Resolution(
            term=term,
            reading=reading,
            resolver="pyopenjtalk",
            confidence=_confidence(1, reading),
            source="pyopenjtalk.g2p",
            metadata={"raw": raw},
        )

    return resolve


def _sudachi_backend() -> Callable[[str], Resolution]:
    dictionary = importlib.import_module("sudachipy.dictionary")
    tokenizer_module = importlib.import_module("sudachipy.tokenizer")
    tokenizer = dictionary.Dictionary().create()
    mode = tokenizer_module.Tokenizer.SplitMode.C

    def resolve(term: str) -> Resolution:
        tokens = list(tokenizer.tokenize(term, mode))
        readings = [t.reading_form() for t in tokens if t.reading_form()]
        raw = "".join(readings)
        reading = normalize_reading(raw)
        return Resolution(
            term=term,
            reading=reading,
            resolver="sudachipy",
            confidence=_confidence(len(tokens), reading),
            source="SudachiPy SplitMode.C",
            metadata={"tokens": [t.surface() for t in tokens], "raw": raw},
        )

    return resolve


def _fugashi_backend() -> Callable[[str], Resolution]:
    fugashi = importlib.import_module("fugashi")
    tagger = fugashi.Tagger()

    def token_reading(token: object) -> str:
        feature = getattr(token, "feature", None)
        for attr in ("kana", "pron", "reading"):
            value = getattr(feature, attr, None)
            if value and value != "*":
                return str(value)
        if isinstance(feature, (tuple, list)):
            for value in reversed(feature):
                if value and value != "*":
                    text = str(value)
                    if any("ァ" <= char <= "ヶ" for char in text):
                        return text
        return ""

    def resolve(term: str) -> Resolution:
        tokens = list(tagger(term))
        readings = [token_reading(t) for t in tokens]
        raw = "".join(r for r in readings if r)
        unknown = any(not r for r in readings)
        reading = normalize_reading(raw)
        return Resolution(
            term=term,
            reading=reading,
            resolver="fugashi_unidic",
            confidence=_confidence(len(tokens), reading, unknown=unknown),
            source="fugashi/UniDic",
            metadata={"tokens": [str(t) for t in tokens], "raw": raw, "unknown": unknown},
        )

    return resolve


BACKENDS: dict[str, tuple[Callable[[], Callable[[str], Resolution]], str, tuple[str, ...]]] = {
    "pyopenjtalk": (_pyopenjtalk_backend, "python -m pip install pyopenjtalk", ("pyopenjtalk",)),
    "sudachipy": (
        _sudachi_backend,
        "python -m pip install sudachipy SudachiDict-core",
        ("sudachipy", "SudachiDict-core"),
    ),
    "fugashi_unidic": (
        _fugashi_backend,
        "python -m pip install fugashi unidic-lite",
        ("fugashi", "unidic-lite"),
    ),
}


def build_available_dict_resolvers() -> tuple[list[DictResolver], list[BackendStatus]]:
    resolvers: list[DictResolver] = []
    statuses: list[BackendStatus] = []
    for name, (factory, install, _packages) in BACKENDS.items():
        try:
            resolvers.append(DictResolver(name, factory()))
            statuses.append(BackendStatus(name, True, "available", install))
        except Exception as exc:  # pragma: no cover - depends on local optional packages
            message = f"skipped: {exc.__class__.__name__}: {exc}"
            LOGGER.warning("Dictionary resolver %s unavailable: %s", name, message)
            statuses.append(BackendStatus(name, False, message, install))
    return resolvers, statuses
