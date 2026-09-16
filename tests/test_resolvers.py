import json
from pathlib import Path

from resolvers.base import Resolution, Resolver
from resolvers.hybrid import HybridResolver, SmartHybridResolver
from resolvers.llm import LlmResolver, clean_llm_output


class StubResolver(Resolver):
    def __init__(self, name, reading, confidence="unknown"):
        super().__init__(name)
        self.reading = reading
        self.confidence = confidence

    def resolve_details(self, term):
        return Resolution(term=term, reading=self.reading, resolver=self.name, confidence=self.confidence)


def test_resolver_contract_returns_normalized_hiragana():
    resolver = StubResolver("stub", "シンキンコウソク")
    assert resolver.resolve("心筋梗塞") == "しんきんこうそく"


def test_llm_resolver_reads_saved_all_results(tmp_path: Path):
    path = tmp_path / "e2b_neweval.json"
    path.write_text(
        json.dumps({"all_results": [{"term": "心筋梗塞", "llm_output": "しんきんこうそく<turn|>"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    resolver = LlmResolver("gemma_e2b_saved", path)
    assert resolver.resolve("心筋梗塞") == "しんきんこうそく"
    assert resolver.resolve_details("未登録語").confidence == "missing"


def test_clean_llm_output_removes_turn_marker_and_normalizes():
    assert clean_llm_output("シンキンコウソク<turn|>") == "しんきんこうそく"


def test_hybrid_uses_dict_when_high_confidence():
    hybrid = HybridResolver(
        StubResolver("dict", "しんきんこうそく", confidence="high"),
        StubResolver("llm", "しんきんこうそくx", confidence="saved_prediction"),
    )
    result = hybrid.resolve_details("心筋梗塞")
    assert result.reading == "しんきんこうそく"
    assert result.metadata["decision"] == "dict_high_confidence"


def test_hybrid_falls_back_to_llm_when_dict_low_confidence():
    hybrid = HybridResolver(
        StubResolver("dict", "しんきん", confidence="low"),
        StubResolver("llm", "しんきんこうそく", confidence="saved_prediction"),
    )
    result = hybrid.resolve_details("心筋梗塞")
    assert result.reading == "しんきんこうそく"
    assert result.metadata["decision"] == "llm_fallback"


def test_hybrid_marks_agreement_high_confidence():
    hybrid = HybridResolver(
        StubResolver("dict", "シンキンコウソク", confidence="low"),
        StubResolver("llm", "しんきんこうそく", confidence="saved_prediction"),
    )
    result = hybrid.resolve_details("心筋梗塞")
    assert result.confidence == "high_agreement"
    assert result.metadata["agreement"] is True


def test_smart_hybrid_routes_to_llm_for_oov_dictionary():
    hybrid = SmartHybridResolver(
        [StubResolver("dict", "", confidence="unavailable")],
        StubResolver("llm", "\u3057\u3093\u304d\u3093\u3053\u3046\u305d\u304f", confidence="saved_prediction"),
    )
    result = hybrid.resolve_details("\u5fc3\u7b4b\u6897\u585e")
    assert result.reading == "\u3057\u3093\u304d\u3093\u3053\u3046\u305d\u304f"
    assert result.metadata["decision"] == "score_primary_llm"


def test_smart_hybrid_routes_to_dictionary_on_domain_boost_and_agreement():
    hybrid = SmartHybridResolver(
        [
            StubResolver("sudachi", "\u304a\u3067\u3044\u3060\u3063\u3059\u3044", confidence="low"),
            StubResolver("fugashi", "\u304a\u3067\u3044\u3060\u3063\u3059\u3044", confidence="low"),
        ],
        StubResolver("llm", "\u304a\u3067\u3044\u3060\u3064\u3059\u3044", confidence="saved_prediction"),
    )
    result = hybrid.resolve_details("\u6c5a\u6ce5\u8131\u6c34")
    assert result.reading == "\u304a\u3067\u3044\u3060\u3063\u3059\u3044"
    assert result.metadata["decision"] == "score_dict"


def test_smart_hybrid_builds_datsuchisso_from_components():
    class ComponentStub(Resolver):
        def __init__(self):
            super().__init__("component_dict")

        def resolve_details(self, term):
            readings = {
                "\u8131": "\u3060\u3064",
                "\u7a92\u7d20": "\u3061\u3063\u305d",
                "\u8131\u7a92\u7d20": "\u3060\u3063\u3061\u3064\u305d",
            }
            return Resolution(term=term, reading=readings.get(term, ""), resolver=self.name, confidence="low")

    hybrid = SmartHybridResolver(
        [ComponentStub()],
        StubResolver("llm", "\u3060\u3063\u3057\u3063\u305d", confidence="saved_prediction"),
    )
    result = hybrid.resolve_details("\u8131\u7a92\u7d20")
    assert result.reading == "\u3060\u3063\u3061\u3063\u305d"
    assert result.metadata["decision"] == "score_dict"
    assert result.metadata["best_dict_source"] == "component_compound"
 
