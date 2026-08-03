import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.generate_training_sentences import (
    Annotation,
    CorpusError,
    SentenceLimits,
    atomic_write_jsonl,
    build_control_records,
    build_generation_slots,
    build_messages,
    generate_group,
    load_annotations,
    make_text_record,
    normalize_sentence,
    parse_args,
    parse_generation_payload,
    read_jsonl,
    select_prompt_terms,
    stable_seed,
    validate_existing_records,
    validate_sentence,
)


LIMITS = SentenceLimits(min_chars=5, max_chars=100)


def test_load_annotations_uses_bom_tsv(tmp_path: Path):
    path = tmp_path / "annotations.tsv"
    path.write_text(
        "term\treading\n活性汚泥法\tかっせいおでいほう\n"
        "最終沈殿池\tさいしゅうちんでんち\n",
        encoding="utf-8-sig",
    )

    assert load_annotations(path) == [
        Annotation(1, "活性汚泥法", "かっせいおでいほう"),
        Annotation(2, "最終沈殿池", "さいしゅうちんでんち"),
    ]


def test_load_annotations_rejects_exact_duplicate(tmp_path: Path):
    path = tmp_path / "annotations.tsv"
    path.write_text(
        "term\treading\n活性汚泥法\tかっせいおでいほう\n"
        "活性汚泥法\tかっせいおでいほう\n",
        encoding="utf-8",
    )

    with pytest.raises(CorpusError, match="duplicate term"):
        load_annotations(path)


def test_load_annotations_rejects_conflicting_duplicate(tmp_path: Path):
    path = tmp_path / "annotations.tsv"
    path.write_text(
        "term\treading\n活性汚泥法\tかっせいおでいほう\n活性汚泥法\t別の読み\n",
        encoding="utf-8",
    )

    with pytest.raises(CorpusError, match="conflicting reading"):
        load_annotations(path)


@pytest.mark.parametrize(
    "contents, message",
    [
        ("term\treading\n\tよみ\n", "non-empty"),
        ("term\treading\n用語\t\n", "non-empty"),
        ("term,reading\n用語,よみ\n", "header"),
        ("term\treading\textra\n用語\tよみ\textra\n", "header"),
        ("term\treading\n", "no term/reading"),
    ],
)
def test_load_annotations_fails_fast_on_empty_or_malformed_rows(
    tmp_path: Path, contents: str, message: str
):
    path = tmp_path / "annotations.tsv"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(CorpusError, match=message):
        load_annotations(path)


def test_generation_slots_have_stable_ids_and_ratio_distribution():
    annotations = [
        Annotation(1, "用語甲", "ようごこう"),
        Annotation(2, "用語乙", "ようごおつ"),
        Annotation(3, "用語丙", "ようごへい"),
    ]

    slots = build_generation_slots(annotations, sentences_per_term=2, replay_ratio=0.5)

    assert [slot.id for slot in slots] == [
        "term-000001-001",
        "term-000001-002",
        "replay-000001-001",
        "term-000002-001",
        "term-000002-002",
        "replay-000002-001",
        "term-000003-001",
        "term-000003-002",
        "replay-000003-001",
    ]
    assert sum(slot.kind == "term" for slot in slots) == 6
    assert sum(slot.kind == "replay" for slot in slots) == 3


def test_control_records_are_deterministic_top_level_silence_and_noise():
    records = build_control_records(
        textual_record_count=20,
        control_ratio=0.2,
        noise_share=0.5,
        duration_sec=1.25,
        seed=17,
    )

    assert [record["kind"] for record in records] == ["silence", "silence", "noise", "noise"]
    assert all(record["sentence"] == record["tts_text"] == "" for record in records)
    assert all(record["control_duration_sec"] == 1.25 for record in records)
    assert records == build_control_records(20, 0.2, 0.5, 1.25, 17)
    assert len({record["control_seed"] for record in records}) == len(records)


def test_term_record_replaces_only_the_authoritative_source_term():
    slot = build_generation_slots(
        [Annotation(1, "活性汚泥法", "かっせいおでいほう")], 1, 0
    )[0]

    record = make_text_record(slot, "担当者が活性汚泥法の運転状態を確認します。")

    assert record["sentence"] == "担当者が活性汚泥法の運転状態を確認します。"
    assert record["tts_text"] == "担当者がかっせいおでいほうの運転状態を確認します。"
    assert record["reading"] == "かっせいおでいほう"


def test_positive_validation_requires_exactly_one_source_term():
    sentence = "活性汚泥法を確認し、活性汚泥法の状態を記録します。"

    reasons = validate_sentence(
        sentence,
        limits=LIMITS,
        required_term="活性汚泥法",
    )

    assert "required_term_count" in reasons


def test_positive_rejects_source_term_when_it_is_part_of_a_longer_annotation():
    sentence = "担当者が膜分離活性汚泥法の運転状態を確認します。"

    reasons = validate_sentence(
        sentence,
        limits=LIMITS,
        required_term="活性汚泥法",
        forbidden_terms=["膜分離活性汚泥法"],
    )

    assert "forbidden_term" in reasons


def test_positive_allows_shorter_annotation_nested_inside_required_long_term():
    sentence = "担当者が膜分離活性汚泥法の運転状態を確認します。"

    reasons = validate_sentence(
        sentence,
        limits=LIMITS,
        required_term="膜分離活性汚泥法",
        forbidden_terms=["活性汚泥法"],
    )

    assert "required_term_count" not in reasons
    assert "forbidden_term" not in reasons


def test_positive_allows_digits_only_inside_authoritative_term():
    accepted = validate_sentence(
        "担当者が5G基地局の運転状態を丁寧に確認します。",
        limits=LIMITS,
        required_term="5G基地局",
    )
    rejected = validate_sentence(
        "担当者が5G基地局を二十四時間ではなく24時間確認します。",
        limits=LIMITS,
        required_term="5G基地局",
    )

    assert "number" not in accepted
    assert "number" in rejected


@pytest.mark.parametrize("term", ["活性汚泥法", "膜分離活性汚泥法", "最終沈殿池"])
def test_replay_validation_excludes_every_annotation_term(term: str):
    sentence = f"担当者が{term}の運転状態を丁寧に確認します。"

    reasons = validate_sentence(
        sentence,
        limits=LIMITS,
        forbidden_terms=["活性汚泥法", "膜分離活性汚泥法", "最終沈殿池"],
    )

    assert "forbidden_term" in reasons


@pytest.mark.parametrize(
    "sentence, reason",
    [
        ("担当者が設備を確認します。\n異常はありません。", "line_break_or_tab"),
        ("担当者が設備を確認します。異常はありません。", "not_one_sentence"),
        ("短い。", "length"),
        ("担当者は設備の詳細を https://example.com で確認します。", "markup"),
        ("担当者が**重要な設備**の状態を確認します。", "markup"),
        ("担当者が三つ目の設備３台を確認します。", "number"),
        ("担当者が確認確認確認を行います。", "repetition"),
        ("担当者が設備を確認します。 ", "empty_or_outer_whitespace"),
    ],
)
def test_strict_sentence_validation(sentence: str, reason: str):
    reasons = validate_sentence(sentence, limits=SentenceLimits(8, 100))
    assert reason in reasons


def test_duplicate_detection_ignores_width_whitespace_and_punctuation():
    original = "担当者が設備の運転状態を確認します。"
    variant = "担当者が 設備の運転状態を確認します！"

    assert normalize_sentence(original) == normalize_sentence(variant)
    assert "duplicate" in validate_sentence(
        variant,
        limits=LIMITS,
        seen_normalized={normalize_sentence(original)},
    )


def test_parse_generation_payload_requires_string_arrays():
    assert parse_generation_payload(
        '{"term_sentences":["正例です。"],"replay_sentences":["対照例です。"]}'
    ) == (["正例です。"], ["対照例です。"])

    with pytest.raises(CorpusError, match="string array"):
        parse_generation_payload('{"term_sentences":"正例です。"}')
    with pytest.raises(CorpusError, match="JSON object"):
        parse_generation_payload("```json\n{}\n```")


def test_atomic_jsonl_round_trip_replaces_old_content(tmp_path: Path):
    path = tmp_path / "corpus.jsonl"
    path.write_text("stale\n", encoding="utf-8")
    records = [{"id": "一", "sentence": "日本語です。"}, {"id": "二", "sentence": "別文です。"}]

    atomic_write_jsonl(path, records)

    assert read_jsonl(path) == records
    assert not list(tmp_path.glob(".corpus.jsonl.*.tmp"))


def test_resume_validation_rejects_tts_text_not_derived_from_tsv():
    annotations = [Annotation(1, "活性汚泥法", "かっせいおでいほう")]
    slots = build_generation_slots(annotations, 1, 0)
    record = make_text_record(slots[0], "担当者が活性汚泥法の状態を確認します。")
    record["tts_text"] = "モデルが勝手に作った読みです。"

    with pytest.raises(CorpusError, match="annotations.tsv"):
        validate_existing_records(
            [record], slots, [], [annotation.term for annotation in annotations], LIMITS
        )


def test_resume_validation_rejects_duplicate_sentences():
    annotations = [Annotation(1, "活性汚泥法", "かっせいおでいほう")]
    slots = build_generation_slots(annotations, 2, 0)
    sentence = "担当者が活性汚泥法の状態を確認します。"
    records = [make_text_record(slot, sentence) for slot in slots]

    with pytest.raises(CorpusError, match="duplicate"):
        validate_existing_records(records, slots, [], ["活性汚泥法"], LIMITS)


def test_seed_is_stable_and_changes_by_slot():
    assert stable_seed(42, "term", 1) == stable_seed(42, "term", 1)
    assert stable_seed(42, "term", 1) != stable_seed(42, "term", 2)
    assert 0 <= stable_seed(42, "term", 1) <= 0x7FFFFFFF


def test_prompt_term_subset_prioritizes_containment_and_is_bounded_and_stable():
    terms = [
        "無関係な用語甲",
        "活性汚泥法",
        "膜分離活性汚泥法",
        "汚泥法",
        "無関係な用語乙",
        "無関係な用語丙",
    ]

    selected = select_prompt_terms(terms, "活性汚泥法", max_terms=4, max_chars=100)
    shuffled = select_prompt_terms(list(reversed(terms)), "活性汚泥法", 4, 100)

    assert selected == shuffled
    assert selected[:3] == ["活性汚泥法", "膜分離活性汚泥法", "汚泥法"]
    assert len(selected) == 4
    assert len(json.dumps(selected, ensure_ascii=False, separators=(",", ":"))) <= 100


def test_prompt_is_domain_agnostic_unless_hint_is_given():
    annotation = Annotation(1, "専門用語", "せんもんようご")

    generic = build_messages(annotation, ["専門用語"], 1, 1, LIMITS, 2)
    hinted = build_messages(
        annotation, ["専門用語"], 1, 1, LIMITS, 2, domain_hint="医療機器"
    )

    assert "下水処理" not in generic[1]["content"]
    assert "対象語が実際に使われる専門分野" in generic[1]["content"]
    assert "分野ヒント: 医療機器" in hinted[1]["content"]


def test_qwen_request_disables_thinking_and_uses_repetition_sampling_controls():
    class FakeCompletions:
        def __init__(self):
            self.kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs
            content = json.dumps(
                {
                    "term_sentences": [
                        "担当者が活性汚泥法をかっせいおでいほうと読み上げます。",
                        "担当者が活性汚泥法の運転状態を確認します。",
                    ],
                    "replay_sentences": [],
                },
                ensure_ascii=False,
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    annotation = Annotation(1, "活性汚泥法", "かっせいおでいほう")
    slot = build_generation_slots([annotation], 1, 0)[0]

    result = asyncio.run(
        generate_group(
            client=client,
            semaphore=asyncio.Semaphore(1),
            annotation=annotation,
            term_slots=[slot],
            replay_slots=[],
            all_terms=[annotation.term],
            seen=set(),
            seen_lock=asyncio.Lock(),
            model="Qwen/Qwen3.6-27B",
            limits=LIMITS,
            max_retries=0,
            retry_backoff_sec=0,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            presence_penalty=1.5,
            max_tokens=512,
            oversample=1,
            seed=42,
            domain_hint="",
            prompt_max_terms=200,
            prompt_max_chars=1200,
        )
    )

    assert not result.missing_term_ids
    assert result.records[0]["sentence"] == "担当者が活性汚泥法の運転状態を確認します。"
    assert result.rejection_reasons["forbidden_term"] == 1
    assert completions.kwargs["response_format"] == {"type": "json_object"}
    assert completions.kwargs["presence_penalty"] == 1.5
    assert completions.kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False},
        "top_k": 20,
    }


def test_cli_defaults_to_resume_and_derives_report_paths(tmp_path: Path):
    output = tmp_path / "training.jsonl"

    args = parse_args(["--out", str(output)])

    assert args.resume is True
    assert args.summary_out == output.with_suffix(".jsonl.summary.json")
    assert args.failures_out == output.with_suffix(".jsonl.failures.jsonl")
    assert (args.top_p, args.top_k, args.presence_penalty) == (0.8, 20, 1.5)
