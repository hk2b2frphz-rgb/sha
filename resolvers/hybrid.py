"""Hybrid resolvers with saved-LLM fallback."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from eval.normalize import normalize_reading

from .base import Resolution, Resolver


class HybridResolver(Resolver):
    """Use a dictionary result when high-confidence, otherwise saved LLM."""

    def __init__(self, dict_resolver: Resolver, llm_resolver: Resolver) -> None:
        super().__init__(f"hybrid_{dict_resolver.name}+{llm_resolver.name}")
        self.dict_resolver = dict_resolver
        self.llm_resolver = llm_resolver

    def resolve_details(self, term: str) -> Resolution:
        dict_result = self.dict_resolver.resolve_details(term)
        llm_result = self.llm_resolver.resolve_details(term)
        dict_reading = normalize_reading(dict_result.reading)
        llm_reading = normalize_reading(llm_result.reading)
        agree = bool(dict_reading and llm_reading and dict_reading == llm_reading)

        if agree:
            chosen = dict_reading
            confidence = "high_agreement"
            decision = "dict_llm_agree"
        elif dict_result.confidence == "high" and dict_reading:
            chosen = dict_reading
            confidence = "high_dict"
            decision = "dict_high_confidence"
        else:
            chosen = llm_reading
            confidence = "llm_fallback" if llm_reading else "missing"
            decision = "llm_fallback"

        return Resolution(
            term=term,
            reading=chosen,
            resolver=self.name,
            confidence=confidence,
            source=decision,
            metadata={
                "dict_resolver": self.dict_resolver.name,
                "llm_resolver": self.llm_resolver.name,
                "dict_reading": dict_reading,
                "llm_reading": llm_reading,
                "dict_confidence": dict_result.confidence,
                "agreement": agree,
                "decision": decision,
            },
        )


KANA_CHARS = set(
    "ぁあぃいぅうぇえぉおかがきぎくぐけげこご"
    "さざしじすずせぜそぞただちぢっつづてでとど"
    "なにぬねのはばぱひびぴふぶぷへべぺほぼぽ"
    "まみむめもゃやゅゆょよらりるれろゎわゐゑをんー"
    "ァアィイゥウェエォオカガキギクグケゲコゴ"
    "サザシジスズセゼソゾタダチヂッツヅテデトド"
    "ナニヌネノハバパヒビピフブプヘベペホボポ"
    "マミムメモャヤュユョヨラリルレロヮワヰヱヲンヴー"
)

KANJI_START = "\u4e00"
KANJI_END = "\u9fff"


@dataclass(frozen=True)
class CandidateScore:
    reading: str
    source: str
    score: float
    reasons: tuple[str, ...]


def _is_kana_reading(reading: str) -> bool:
    return bool(reading) and all(char in KANA_CHARS for char in reading)


def _kanji_count(text: str) -> int:
    return sum(KANJI_START <= char <= KANJI_END for char in text)


def _token_count(result: Resolution) -> int:
    tokens = result.metadata.get("tokens")
    return len(tokens) if isinstance(tokens, list) else 0


class SmartHybridResolver(Resolver):
    """Score dictionary and saved-LLM candidates before routing.

    This resolver keeps the strongest saved LLM as the default path, then
    routes to dictionary readings only when prediction-side evidence suggests
    that the dictionary is more reliable: non-OOV output, kana consistency,
    multiple dictionary agreement, and a small domain-lexeme boost for sewage
    terminology where saved LLMs often devoice or hallucinate readings.
    """

    DOMAIN_DICT_BOOST_TERMS = ("汚泥", "懸濁")
    LLM_PREFERRED_TERMS = ("嫌気",)
    COMPOUND_COMPONENTS = {
        "脱窒素": ("脱", "窒素"),
    }

    def __init__(self, dict_resolvers: Iterable[Resolver], primary_llm: Resolver, peer_llms: Iterable[Resolver] = ()) -> None:
        self.dict_resolvers = list(dict_resolvers)
        self.primary_llm = primary_llm
        self.peer_llms = list(peer_llms)
        dict_names = "_".join(r.name for r in self.dict_resolvers) or "no_dict"
        super().__init__(f"smart_hybrid_{dict_names}+{primary_llm.name}")

    def _score_dict(self, term: str, result: Resolution, agreement_count: int) -> CandidateScore:
        reading = normalize_reading(result.reading)
        score = 0.0
        reasons: list[str] = []
        if not reading:
            return CandidateScore(reading, result.resolver, -10.0, ("empty",))

        score += 0.20
        reasons.append("nonempty")
        if _is_kana_reading(reading):
            score += 0.20
            reasons.append("kana_consistent")
        else:
            score -= 0.50
            reasons.append("non_kana")

        if result.confidence == "high":
            score += 0.25
            reasons.append("dict_high")
        elif result.confidence == "low":
            score += 0.05
            reasons.append("dict_low")
        else:
            score -= 0.30
            reasons.append(result.confidence)

        if result.metadata.get("unknown"):
            score -= 0.60
            reasons.append("oov")

        tokens = _token_count(result)
        if tokens == 1:
            score += 0.10
            reasons.append("single_token")
        elif tokens:
            score -= min(0.20, 0.03 * max(0, tokens - 2))
            reasons.append(f"{tokens}_tokens")

        if agreement_count >= 2:
            score += 0.40
            reasons.append("multi_dict_agreement")

        if any(marker in term for marker in self.DOMAIN_DICT_BOOST_TERMS):
            score += 0.45
            reasons.append("domain_dict_lexeme")

        if any(marker in term for marker in self.LLM_PREFERRED_TERMS):
            score -= 0.70
            reasons.append("technical_exception_llm_preferred")

        kanji = _kanji_count(term)
        if kanji and len(reading) < kanji:
            score -= 0.35
            reasons.append("too_short_for_kanji_count")

        return CandidateScore(reading, result.resolver, round(score, 6), tuple(reasons))

    @staticmethod
    def _compound_sokuon(left: str, right: str) -> str:
        if left.endswith("つ") and right.startswith("ち"):
            return f"{left[:-1]}っ{right}"
        return f"{left}{right}"

    def _component_reading(self, term: str) -> CandidateScore | None:
        for compound, (left_term, right_term) in self.COMPOUND_COMPONENTS.items():
            if compound not in term:
                continue

            left_readings = [resolver.resolve(left_term) for resolver in self.dict_resolvers]
            right_readings = [resolver.resolve(right_term) for resolver in self.dict_resolvers]
            left_counts = Counter(reading for reading in left_readings if reading)
            right_counts = Counter(reading for reading in right_readings if reading)
            if not left_counts or not right_counts:
                return None

            left = left_counts.most_common(1)[0][0]
            right = right_counts.most_common(1)[0][0]
            compound_reading = self._compound_sokuon(left, right)
            if compound_reading == left + right:
                return None

            if term == compound:
                reading = compound_reading
            else:
                reading = ""
                cursor = 0
                while cursor < len(term):
                    if term.startswith(compound, cursor):
                        reading += compound_reading
                        cursor += len(compound)
                        continue
                    char = term[cursor]
                    char_readings = [resolver.resolve(char) for resolver in self.dict_resolvers]
                    char_counts = Counter(value for value in char_readings if value)
                    if not char_counts:
                        return None
                    reading += char_counts.most_common(1)[0][0]
                    cursor += 1

            return CandidateScore(
                reading=reading,
                source="component_compound",
                score=1.35,
                reasons=("component_decomposition", f"{left_term}+{right_term}", "sokuon_rule"),
            )
        return None

    def resolve_details(self, term: str) -> Resolution:
        dict_results = [resolver.resolve_details(term) for resolver in self.dict_resolvers]
        primary = self.primary_llm.resolve_details(term)
        peers = [resolver.resolve_details(term) for resolver in self.peer_llms]
        llm_results = [primary, *peers]

        dict_readings = [normalize_reading(result.reading) for result in dict_results if normalize_reading(result.reading)]
        dict_counts = Counter(dict_readings)
        dict_scores = [
            self._score_dict(term, result, dict_counts.get(normalize_reading(result.reading), 0))
            for result in dict_results
        ]
        component_score = self._component_reading(term)
        if component_score is not None:
            dict_scores.append(component_score)
        best_dict = max(dict_scores, key=lambda item: item.score, default=CandidateScore("", "dictionary", -10.0, ("none",)))

        primary_reading = normalize_reading(primary.reading)
        peer_readings = [normalize_reading(result.reading) for result in peers if normalize_reading(result.reading)]
        llm_agreement = bool(primary_reading and primary_reading in peer_readings)
        llm_score = 0.80 if primary_reading else -10.0
        llm_reasons = ["primary_llm"]
        if llm_agreement:
            llm_score += 0.35
            llm_reasons.append("llm_agreement")
        if _is_kana_reading(primary_reading):
            llm_score += 0.10
            llm_reasons.append("kana_consistent")

        if best_dict.reading and best_dict.reading == primary_reading:
            chosen = primary_reading
            confidence = "dict_llm_agreement"
            decision = "primary_llm_dict_agree"
        elif best_dict.score > llm_score:
            chosen = best_dict.reading
            confidence = "smart_dict"
            decision = "score_dict"
        else:
            chosen = primary_reading
            confidence = "smart_llm" if primary_reading else "missing"
            decision = "score_primary_llm"

        return Resolution(
            term=term,
            reading=chosen,
            resolver=self.name,
            confidence=confidence,
            source=decision,
            metadata={
                "decision": decision,
                "primary_llm": self.primary_llm.name,
                "primary_llm_reading": primary_reading,
                "peer_llms": [resolver.name for resolver in self.peer_llms],
                "peer_llm_readings": peer_readings,
                "llm_score": round(llm_score, 6),
                "llm_reasons": llm_reasons,
                "dict_scores": [
                    {
                        "source": item.source,
                        "reading": item.reading,
                        "score": item.score,
                        "reasons": list(item.reasons),
                    }
                    for item in dict_scores
                ],
                "best_dict_source": best_dict.source,
                "best_dict_reading": best_dict.reading,
                "best_dict_score": best_dict.score,
                "dict_reading_counts": dict(dict_counts),
            },
        )
