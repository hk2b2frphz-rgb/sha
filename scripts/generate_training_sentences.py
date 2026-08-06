#!/usr/bin/env python3
"""Generate a guarded ASR adaptation corpus through a vLLM chat endpoint.

``annotations.tsv`` is the sole authority for both the written technical term and
its TTS reading.  The language model only writes surrounding sentences; readings
returned by a model are never consumed.

The output is checkpointed with an atomic replace after every completed source
term.  Re-running the same command resumes missing stable IDs by default.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import math
import os
import re
import sys
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


LOGGER = logging.getLogger("generate_training_sentences")
DEFAULT_MODEL = "Qwen/Qwen3.6-27B"
TERMINATORS = frozenset("。！？!?")
MARKUP_RE = re.compile(
    r"(?:https?://|www\.|```|`|<[^>]*>|\[[^\]]*\](?:\([^)]*\))?|"
    r"\*\*|__|~~|(?:^|\s)[#>*+-]\s|\{[^}]*\})",
    re.IGNORECASE,
)


class CorpusError(ValueError):
    """Raised when authoritative input or resumable output is inconsistent."""


@dataclass(frozen=True, slots=True)
class Annotation:
    source_index: int
    term: str
    reading: str


@dataclass(frozen=True, slots=True)
class SentenceLimits:
    min_chars: int = 18
    max_chars: int = 80

    def __post_init__(self) -> None:
        if self.min_chars < 1:
            raise ValueError("min_chars must be positive")
        if self.max_chars < self.min_chars:
            raise ValueError("max_chars must be greater than or equal to min_chars")


@dataclass(frozen=True, slots=True)
class GenerationSlot:
    id: str
    kind: str
    source_term: str
    source_index: int
    term: str
    reading: str


@dataclass(slots=True)
class GroupResult:
    source_index: int
    records: list[dict[str, Any]]
    missing_term_ids: list[str]
    missing_replay_ids: list[str]
    attempts: int
    rejection_reasons: Counter[str]
    api_errors: list[str]


def load_annotations(path: Path) -> list[Annotation]:
    """Load a strict UTF-8(-BOM) ``term,reading`` CSV or TSV mapping.

    Every term must occur once.  Both an exact duplicate and reuse with a
    conflicting reading fail immediately so target quotas remain unambiguous.
    """

    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise CorpusError(f"cannot read annotations: {path}: {exc}") from exc

    with handle:
        lines = handle.readlines()

    header_line = next(
        (
            line
            for line in lines
            if line.strip() and not line.lstrip().startswith("#")
        ),
        "",
    )
    if "\t" in header_line:
        delimiter = "\t"
    elif "," in header_line:
        delimiter = ","
    else:
        raise CorpusError(f"{path}: missing header 'term<TAB>reading' or 'term,reading'")

    annotations: list[Annotation] = []
    by_term: dict[str, tuple[str, int]] = {}
    saw_header = False
    for line_no, columns in enumerate(csv.reader(lines, delimiter=delimiter), 1):
        if not columns or all(not value.strip() for value in columns):
            continue
        if columns[0].lstrip().startswith("#"):
            continue
        if not saw_header:
            normalized = [value.strip().lower() for value in columns]
            if normalized != ["term", "reading"]:
                raise CorpusError(
                    f"{path}:{line_no}: expected exactly the header "
                    "'term<TAB>reading' or 'term,reading'"
                )
            saw_header = True
            continue
        if len(columns) != 2:
            raise CorpusError(f"{path}:{line_no}: expected exactly two columns")
        term, reading = (value.strip() for value in columns)
        if not term or not reading:
            raise CorpusError(f"{path}:{line_no}: term and reading must both be non-empty")
        if any(char in term or char in reading for char in ("\r", "\n", "\t")):
            raise CorpusError(f"{path}:{line_no}: term/reading contains a control separator")
        previous = by_term.get(term)
        if previous is not None:
            previous_reading, previous_line = previous
            if reading != previous_reading:
                raise CorpusError(
                    f"{path}:{line_no}: conflicting reading for {term!r}; "
                    f"line {previous_line} has {previous_reading!r}, got {reading!r}"
                )
            raise CorpusError(
                f"{path}:{line_no}: duplicate term {term!r}; first seen on line {previous_line}"
            )
        source_index = len(annotations) + 1
        annotations.append(Annotation(source_index, term, reading))
        by_term[term] = (reading, line_no)

    if not saw_header:
        raise CorpusError(f"{path}: missing header 'term<TAB>reading' or 'term,reading'")
    if not annotations:
        raise CorpusError(f"{path}: no term/reading rows")
    return annotations


def round_half_up(value: float) -> int:
    """Round a non-negative quota without Python's ties-to-even surprise."""

    if not math.isfinite(value) or value < 0:
        raise ValueError("quota must be a finite non-negative number")
    return int(math.floor(value + 0.5))


def distribute_count(total: int, buckets: int) -> list[int]:
    """Distribute ``total`` deterministically, favoring earlier source rows."""

    if total < 0 or buckets < 1:
        raise ValueError("total must be non-negative and buckets must be positive")
    base, remainder = divmod(total, buckets)
    return [base + (index < remainder) for index in range(buckets)]


def build_generation_slots(
    annotations: Sequence[Annotation],
    sentences_per_term: int,
    replay_ratio: float,
) -> list[GenerationSlot]:
    """Create deterministic term and replay targets.

    ``replay_ratio`` is measured against the number of positive term sentences.
    Each replay slot retains a source term solely as a domain prompt anchor.
    """

    if not annotations:
        raise ValueError("annotations must not be empty")
    if sentences_per_term < 1:
        raise ValueError("sentences_per_term must be positive")
    positive_total = len(annotations) * sentences_per_term
    replay_total = round_half_up(positive_total * replay_ratio)
    replay_by_source = distribute_count(replay_total, len(annotations))
    slots: list[GenerationSlot] = []
    for annotation, replay_count in zip(annotations, replay_by_source, strict=True):
        for ordinal in range(1, sentences_per_term + 1):
            slots.append(
                GenerationSlot(
                    id=f"term-{annotation.source_index:06d}-{ordinal:03d}",
                    kind="term",
                    source_term=annotation.term,
                    source_index=annotation.source_index,
                    term=annotation.term,
                    reading=annotation.reading,
                )
            )
        for ordinal in range(1, replay_count + 1):
            slots.append(
                GenerationSlot(
                    id=f"replay-{annotation.source_index:06d}-{ordinal:03d}",
                    kind="replay",
                    source_term=annotation.term,
                    source_index=annotation.source_index,
                    term="",
                    reading="",
                )
            )
    return slots


def stable_seed(seed: int, *parts: object) -> int:
    payload = "\0".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    # vLLM/OpenAI-compatible servers generally expect a signed 32-bit seed.
    return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "big") & 0x7FFFFFFF


def select_prompt_terms(
    all_terms: Sequence[str],
    source_term: str,
    max_terms: int,
    max_chars: int,
) -> list[str]:
    """Select a bounded, deterministic prompt subset of authoritative terms.

    Local validation still checks every annotation.  The prompt subset always
    prioritizes the source and its substring/superstring relations, then uses a
    content-hash order so adding an unrelated annotation causes minimal churn.
    """

    if max_terms < 1 or max_chars < 1:
        raise ValueError("prompt term and character limits must be positive")
    unique_terms = list(dict.fromkeys(term for term in all_terms if term))
    if source_term not in unique_terms:
        unique_terms.insert(0, source_term)

    related = [
        term
        for term in unique_terms
        if term != source_term and (term in source_term or source_term in term)
    ]
    # Longer containing terms are most important: they prevent accepting a short
    # source term merely because it appears inside another dictionary entry.
    related.sort(key=lambda term: (source_term not in term, -len(term), term))
    unrelated = [term for term in unique_terms if term != source_term and term not in related]
    unrelated.sort(
        key=lambda term: (
            hashlib.blake2s(f"{source_term}\0{term}".encode("utf-8"), digest_size=8).digest(),
            term,
        )
    )

    selected: list[str] = []
    for term in [source_term, *related, *unrelated]:
        if len(selected) >= max_terms:
            break
        candidate = [*selected, term]
        serialized_chars = len(json.dumps(candidate, ensure_ascii=False, separators=(",", ":")))
        if serialized_chars <= max_chars or not selected:
            selected.append(term)
    return selected


def build_control_records(
    textual_record_count: int,
    control_ratio: float,
    noise_share: float,
    duration_sec: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Build deterministic silence/noise no-transcript controls."""

    if textual_record_count < 0:
        raise ValueError("textual_record_count must be non-negative")
    if not 0 <= noise_share <= 1:
        raise ValueError("noise_share must be between zero and one")
    if not math.isfinite(duration_sec) or duration_sec <= 0:
        raise ValueError("duration_sec must be positive and finite")
    control_total = round_half_up(textual_record_count * control_ratio)
    noise_count = round_half_up(control_total * noise_share)
    silence_count = control_total - noise_count
    records: list[dict[str, Any]] = []
    for kind, count in (("silence", silence_count), ("noise", noise_count)):
        for ordinal in range(1, count + 1):
            records.append(
                {
                    "id": f"{kind}-{ordinal:06d}",
                    "kind": kind,
                    "source_term": "",
                    "source_index": ordinal,
                    "term": "",
                    "reading": "",
                    "sentence": "",
                    "tts_text": "",
                    "control_duration_sec": duration_sec,
                    "control_seed": stable_seed(seed, kind, ordinal),
                }
            )
    return records


def normalize_sentence(text: str) -> str:
    """Canonical form used only for conservative duplicate detection."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        char
        for char in normalized
        if not char.isspace() and not unicodedata.category(char).startswith(("P", "S"))
    )


def has_suspicious_repetition(text: str) -> bool:
    """Detect common autoregressive loops while tolerating normal Japanese prose."""

    body = text[:-1] if text and text[-1] in TERMINATORS else text
    compact = re.sub(r"\s+", "", body)
    if re.search(r"(.)\1{2,}", compact):
        return True
    # Long phrases repeated twice, or shorter chunks repeated three times, are
    # overwhelmingly generation loops in the short utterances accepted here.
    if re.search(r"(.{4,20})\1", compact):
        return True
    if re.search(r"(.{2,12})\1{2,}", compact):
        return True
    clauses = [normalize_sentence(part) for part in re.split(r"[、，,;；]", body)]
    nonempty_clauses = [part for part in clauses if len(part) >= 3]
    return len(nonempty_clauses) != len(set(nonempty_clauses))


def _term_occurrences(text: str, term: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        index = text.find(term, start)
        if index < 0:
            return spans
        spans.append((index, index + len(term)))
        start = index + max(1, len(term))


def _forbidden_outside_required(
    sentence: str,
    forbidden_terms: Iterable[str],
    required_span: tuple[int, int] | None,
) -> bool:
    for term in forbidden_terms:
        if not term:
            continue
        for start, end in _term_occurrences(sentence, term):
            # A shorter annotation nested inside the one required occurrence is
            # not an additional technical term (e.g. 活性汚泥法 in 膜分離活性汚泥法).
            if required_span and required_span[0] <= start and end <= required_span[1]:
                continue
            return True
    return False


def validate_sentence(
    sentence: object,
    *,
    limits: SentenceLimits,
    required_term: str | None = None,
    forbidden_terms: Iterable[str] = (),
    seen_normalized: Iterable[str] = (),
) -> list[str]:
    """Return all rejection reasons for a generated sentence."""

    reasons: list[str] = []
    if not isinstance(sentence, str):
        return ["not_string"]
    if not sentence or sentence != sentence.strip():
        reasons.append("empty_or_outer_whitespace")
    if any(char in sentence for char in "\r\n\t\u2028\u2029"):
        reasons.append("line_break_or_tab")
    char_count = len(sentence)
    if not limits.min_chars <= char_count <= limits.max_chars:
        reasons.append("length")
    terminators = [index for index, char in enumerate(sentence) if char in TERMINATORS]
    if len(terminators) != 1 or not sentence or terminators[0] != len(sentence) - 1:
        reasons.append("not_one_sentence")
    if MARKUP_RE.search(sentence):
        reasons.append("markup")
    # Product names and standards can legitimately contain digits (5G,
    # ISO9001, COVID-19).  Permit digits inside the one authoritative target,
    # while still rejecting model-invented numbers in the surrounding prose.
    content_without_target = (
        sentence.replace(required_term, "対象語", 1) if required_term else sentence
    )
    if any(
        unicodedata.category(char).startswith("N") for char in content_without_target
    ):
        reasons.append("number")
    repetition_text = content_without_target
    if has_suspicious_repetition(repetition_text):
        reasons.append("repetition")

    required_span: tuple[int, int] | None = None
    if required_term is not None:
        occurrences = _term_occurrences(sentence, required_term)
        if len(occurrences) != 1:
            reasons.append("required_term_count")
        else:
            required_span = occurrences[0]
    if _forbidden_outside_required(sentence, forbidden_terms, required_span):
        reasons.append("forbidden_term")

    normalized = normalize_sentence(sentence)
    if not normalized:
        reasons.append("empty_normalized")
    elif normalized in set(seen_normalized):
        reasons.append("duplicate")
    return reasons


def make_text_record(slot: GenerationSlot, sentence: str) -> dict[str, Any]:
    if slot.kind == "term":
        if sentence.count(slot.term) != 1:
            raise CorpusError(f"{slot.id}: target term must occur exactly once")
        tts_text = sentence.replace(slot.term, slot.reading)
    elif slot.kind == "replay":
        tts_text = sentence
    else:
        raise CorpusError(f"{slot.id}: unsupported text kind: {slot.kind}")
    return {
        "id": slot.id,
        "kind": slot.kind,
        "source_term": slot.source_term,
        "source_index": slot.source_index,
        "term": slot.term,
        "reading": slot.reading,
        "sentence": sentence,
        "tts_text": tts_text,
    }


def parse_generation_payload(content: str) -> tuple[list[str], list[str]]:
    """Parse the exact JSON object requested from vLLM."""

    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError) as exc:
        raise CorpusError(f"response is not a JSON object: {exc}") from exc
    if not isinstance(payload, dict):
        raise CorpusError("response JSON must be an object")

    def get_list(name: str) -> list[str]:
        value = payload.get(name, [])
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise CorpusError(f"response field {name!r} must be a string array")
        return value

    return get_list("term_sentences"), get_list("replay_sentences")


def build_messages(
    annotation: Annotation,
    prompt_terms: Sequence[str],
    term_count: int,
    replay_count: int,
    limits: SentenceLimits,
    oversample: int,
    domain_hint: str = "",
) -> list[dict[str, str]]:
    forbidden_json = json.dumps(list(prompt_terms), ensure_ascii=False, separators=(",", ":"))
    domain_instruction = (
        f"分野ヒント: {domain_hint.strip()}"
        if domain_hint.strip()
        else "対象語が実際に使われる専門分野と利用場面を文脈にしてください。"
    )
    system = (
        "あなたは日本語の音声認識学習コーパスを作る編集者です。"
        "回答は指定されたJSON objectだけにし、解説やMarkdownを出力しません。"
        "思考過程も出力しません。"
    )
    user = f"""対象語に合う専門分野の自然な発話文を作成してください。
{domain_instruction}
対象専門用語: {annotation.term}
禁止語の代表一覧: {forbidden_json}

term_sentences は対象専門用語を表記どおりちょうど一回含む文です。
replay_sentences は同じ分野・会話域の文ですが、禁止語の代表一覧を一つも含めません。
必要候補数は term_sentences={term_count * oversample}、replay_sentences={replay_count * oversample} です。

各文の条件:
- 一文だけで、末尾だけに句点または終止記号を置く
- {limits.min_chars}文字以上{limits.max_chars}文字以下
- 改行、箇条書き、Markdown、HTML、URLを使わず、対象専門用語内を除き数字を使わない
- 同じ語句や節を反復しない
- 候補同士で内容・言い回しを重複させない
- 読み仮名を生成したり、専門用語を括弧で説明したりしない

次の形のJSON objectだけを返してください:
{{"term_sentences":["..."],"replay_sentences":["..."]}}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _message_content(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise CorpusError("chat response has no first message content") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, Mapping):
                pieces.append(str(item.get("text") or item.get("content") or ""))
            else:
                pieces.append(str(getattr(item, "text", "")))
        return "".join(pieces)
    raise CorpusError("chat response content is not text")


async def _reserve_candidate(
    candidate: str,
    *,
    limits: SentenceLimits,
    required_term: str | None,
    forbidden_terms: Iterable[str],
    seen: set[str],
    seen_lock: asyncio.Lock,
) -> list[str]:
    reasons = validate_sentence(
        candidate,
        limits=limits,
        required_term=required_term,
        forbidden_terms=forbidden_terms,
    )
    if reasons:
        return reasons
    normalized = normalize_sentence(candidate)
    async with seen_lock:
        if normalized in seen:
            return ["duplicate"]
        seen.add(normalized)
    return []


async def generate_group(
    *,
    client: Any,
    semaphore: asyncio.Semaphore,
    annotation: Annotation,
    term_slots: Sequence[GenerationSlot],
    replay_slots: Sequence[GenerationSlot],
    all_terms: Sequence[str],
    seen: set[str],
    seen_lock: asyncio.Lock,
    model: str,
    limits: SentenceLimits,
    max_retries: int,
    retry_backoff_sec: float,
    temperature: float,
    top_p: float,
    top_k: int,
    presence_penalty: float,
    max_tokens: int,
    oversample: int,
    seed: int,
    domain_hint: str,
    prompt_max_terms: int,
    prompt_max_chars: int,
) -> GroupResult:
    remaining_term = list(term_slots)
    remaining_replay = list(replay_slots)
    records: list[dict[str, Any]] = []
    rejection_reasons: Counter[str] = Counter()
    api_errors: list[str] = []
    attempts = 0

    for attempt in range(max_retries + 1):
        if not remaining_term and not remaining_replay:
            break
        attempts += 1
        prompt_terms = select_prompt_terms(
            all_terms,
            annotation.term,
            max_terms=prompt_max_terms,
            max_chars=prompt_max_chars,
        )
        messages = build_messages(
            annotation,
            prompt_terms,
            len(remaining_term),
            len(remaining_replay),
            limits,
            oversample,
            domain_hint,
        )
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    max_tokens=max_tokens,
                    seed=stable_seed(seed, annotation.source_index, attempt),
                    response_format={"type": "json_object"},
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False},
                        "top_k": top_k,
                    },
                )
            term_candidates, replay_candidates = parse_generation_payload(_message_content(response))
        except Exception as exc:  # noqa: BLE001 - API SDK exposes many transport subclasses
            message = f"{type(exc).__name__}: {exc}"
            api_errors.append(message)
            LOGGER.warning(
                "source %d (%s), attempt %d/%d failed: %s",
                annotation.source_index,
                annotation.term,
                attempt + 1,
                max_retries + 1,
                message,
            )
            if attempt < max_retries and retry_backoff_sec:
                await asyncio.sleep(retry_backoff_sec * (attempt + 1))
            continue

        for raw_candidate in term_candidates:
            if not remaining_term:
                break
            candidate = raw_candidate
            other_terms = [term for term in all_terms if term != annotation.term]
            # Reject an explicit kana gloss such as "用語（よみ）".  The sole
            # pronunciation authority is annotations.tsv and make_text_record()
            # performs that replacement only after the written target is valid.
            if annotation.reading and annotation.reading != annotation.term:
                other_terms.append(annotation.reading)
            reasons = await _reserve_candidate(
                candidate,
                limits=limits,
                required_term=annotation.term,
                forbidden_terms=other_terms,
                seen=seen,
                seen_lock=seen_lock,
            )
            if reasons:
                rejection_reasons.update(reasons)
                continue
            records.append(make_text_record(remaining_term.pop(0), candidate))

        for raw_candidate in replay_candidates:
            if not remaining_replay:
                break
            candidate = raw_candidate
            reasons = await _reserve_candidate(
                candidate,
                limits=limits,
                required_term=None,
                forbidden_terms=all_terms,
                seen=seen,
                seen_lock=seen_lock,
            )
            if reasons:
                rejection_reasons.update(reasons)
                continue
            records.append(make_text_record(remaining_replay.pop(0), candidate))

        if (remaining_term or remaining_replay) and attempt < max_retries and retry_backoff_sec:
            await asyncio.sleep(retry_backoff_sec * (attempt + 1))

    return GroupResult(
        source_index=annotation.source_index,
        records=records,
        missing_term_ids=[slot.id for slot in remaining_term],
        missing_replay_ids=[slot.id for slot in remaining_replay],
        attempts=attempts,
        rejection_reasons=rejection_reasons,
        api_errors=api_errors,
    )


def _fixed_slot_fields(slot: GenerationSlot) -> dict[str, Any]:
    return {
        "id": slot.id,
        "kind": slot.kind,
        "source_term": slot.source_term,
        "source_index": slot.source_index,
        "term": slot.term,
        "reading": slot.reading,
    }


def validate_existing_records(
    records: Sequence[Mapping[str, Any]],
    slots: Sequence[GenerationSlot],
    controls: Sequence[Mapping[str, Any]],
    all_terms: Sequence[str],
    limits: SentenceLimits,
) -> set[str]:
    """Validate a checkpoint and return its normalized textual sentences."""

    expected: dict[str, Mapping[str, Any]] = {
        slot.id: _fixed_slot_fields(slot) for slot in slots
    }
    expected.update({str(record["id"]): record for record in controls})
    seen_ids: set[str] = set()
    seen_sentences: set[str] = set()
    for record in records:
        record_id = str(record.get("id", ""))
        if not record_id:
            raise CorpusError("resume record has no id")
        if record_id in seen_ids:
            raise CorpusError(f"duplicate resume id: {record_id}")
        seen_ids.add(record_id)
        template = expected.get(record_id)
        if template is None:
            raise CorpusError(f"resume id is not in the current target plan: {record_id}")
        for key in ("kind", "source_term", "source_index", "term", "reading"):
            if record.get(key) != template.get(key):
                raise CorpusError(
                    f"{record_id}: resume field {key!r} differs from current annotations/config"
                )
        kind = str(record.get("kind"))
        if kind in {"silence", "noise"}:
            for key in ("sentence", "tts_text"):
                if record.get(key) != "":
                    raise CorpusError(f"{record_id}: control {key} must be empty")
            for key in ("control_duration_sec", "control_seed"):
                if record.get(key) != template.get(key):
                    raise CorpusError(f"{record_id}: control field {key!r} changed")
            continue

        sentence = record.get("sentence")
        slot = next(slot for slot in slots if slot.id == record_id)
        forbidden = all_terms if kind == "replay" else [
            term for term in all_terms if term != slot.term
        ]
        if kind == "term" and slot.reading and slot.reading != slot.term:
            forbidden = [*forbidden, slot.reading]
        reasons = validate_sentence(
            sentence,
            limits=limits,
            required_term=slot.term if kind == "term" else None,
            forbidden_terms=forbidden,
            seen_normalized=seen_sentences,
        )
        if reasons:
            raise CorpusError(f"{record_id}: invalid resume sentence: {', '.join(reasons)}")
        expected_tts = make_text_record(slot, str(sentence))["tts_text"]
        if record.get("tts_text") != expected_tts:
            raise CorpusError(f"{record_id}: tts_text is not derived from annotations.tsv")
        seen_sentences.add(normalize_sentence(str(sentence)))
    return seen_sentences


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CorpusError(f"cannot read resume output: {path}: {exc}") from exc
    for line_no, raw in enumerate(lines, 1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CorpusError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise CorpusError(f"{path}:{line_no}: each JSONL row must be an object")
        records.append(record)
    return records


def atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        finally:
            raise


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        finally:
            raise


def _ordered_records(
    records_by_id: Mapping[str, Mapping[str, Any]],
    slots: Sequence[GenerationSlot],
    controls: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    ordered_ids = [slot.id for slot in slots] + [str(record["id"]) for record in controls]
    return [records_by_id[record_id] for record_id in ordered_ids if record_id in records_by_id]


async def run_generation(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    annotations = load_annotations(args.annotations)
    limits = SentenceLimits(args.min_chars, args.max_chars)
    slots = build_generation_slots(annotations, args.sentences_per_term, args.replay_ratio)
    controls = build_control_records(
        len(slots), args.control_ratio, args.noise_share, args.control_duration_sec, args.seed
    )
    all_terms = [annotation.term for annotation in annotations]

    existing: list[dict[str, Any]] = []
    if args.resume and args.out.exists():
        existing = read_jsonl(args.out)
    seen = validate_existing_records(existing, slots, controls, all_terms, limits)
    records_by_id: dict[str, dict[str, Any]] = {
        str(record["id"]): dict(record) for record in existing
    }
    for control in controls:
        records_by_id.setdefault(str(control["id"]), dict(control))
    # Checkpoint the validated resume state and deterministic controls before API work.
    atomic_write_jsonl(args.out, _ordered_records(records_by_id, slots, controls))

    slots_by_source: dict[int, list[GenerationSlot]] = {
        annotation.source_index: [] for annotation in annotations
    }
    for slot in slots:
        if slot.id not in records_by_id:
            slots_by_source[slot.source_index].append(slot)
    pending_annotations = [
        annotation for annotation in annotations if slots_by_source[annotation.source_index]
    ]

    results: list[GroupResult] = []
    if pending_annotations:
        try:
            from openai import AsyncOpenAI  # Lazy so pure helpers/tests need no SDK.
        except ImportError as exc:
            raise CorpusError("the 'openai' package is required for generation") from exc
        client = AsyncOpenAI(
            base_url=args.base_url,
            api_key=args.api_key,
            timeout=args.timeout,
            max_retries=0,  # Retries are explicit so failures and seeds are auditable.
        )
        semaphore = asyncio.Semaphore(args.concurrency)
        seen_lock = asyncio.Lock()
        tasks = []
        for annotation in pending_annotations:
            missing = slots_by_source[annotation.source_index]
            tasks.append(
                asyncio.create_task(
                    generate_group(
                        client=client,
                        semaphore=semaphore,
                        annotation=annotation,
                        term_slots=[slot for slot in missing if slot.kind == "term"],
                        replay_slots=[slot for slot in missing if slot.kind == "replay"],
                        all_terms=all_terms,
                        seen=seen,
                        seen_lock=seen_lock,
                        model=args.model,
                        limits=limits,
                        max_retries=args.max_retries,
                        retry_backoff_sec=args.retry_backoff_sec,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        presence_penalty=args.presence_penalty,
                        max_tokens=args.max_tokens,
                        oversample=args.oversample,
                        seed=args.seed,
                        domain_hint=args.domain_hint,
                        prompt_max_terms=args.prompt_max_terms,
                        prompt_max_chars=args.prompt_max_chars,
                    )
                )
            )
        try:
            for completed in asyncio.as_completed(tasks):
                result = await completed
                results.append(result)
                for record in result.records:
                    records_by_id[str(record["id"])] = record
                atomic_write_jsonl(args.out, _ordered_records(records_by_id, slots, controls))
                LOGGER.info(
                    "source %d complete: accepted=%d, missing=%d",
                    result.source_index,
                    len(result.records),
                    len(result.missing_term_ids) + len(result.missing_replay_ids),
                )
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await client.close()

    expected_text_ids = {slot.id for slot in slots}
    missing_ids = sorted(expected_text_ids - records_by_id.keys())
    failures: list[dict[str, Any]] = []
    result_by_source = {result.source_index: result for result in results}
    slot_by_id = {slot.id: slot for slot in slots}
    for missing_id in missing_ids:
        slot = slot_by_id[missing_id]
        result = result_by_source.get(slot.source_index)
        failures.append(
            {
                "id": missing_id,
                "kind": slot.kind,
                "source_term": slot.source_term,
                "source_index": slot.source_index,
                "attempts": result.attempts if result else 0,
                "rejection_reasons": dict(result.rejection_reasons) if result else {},
                "api_errors": result.api_errors if result else [],
            }
        )

    completed_by_kind = Counter(str(record["kind"]) for record in records_by_id.values())
    planned_by_kind = Counter(slot.kind for slot in slots)
    planned_by_kind.update(str(control["kind"]) for control in controls)
    summary: dict[str, Any] = {
        "status": "complete" if not missing_ids else "incomplete",
        "model": args.model,
        "seed": args.seed,
        "annotation_count": len(annotations),
        "planned": sum(planned_by_kind.values()),
        "completed": sum(completed_by_kind.values()),
        "missing": len(missing_ids),
        "planned_by_kind": dict(sorted(planned_by_kind.items())),
        "completed_by_kind": dict(sorted(completed_by_kind.items())),
        "output": str(args.out),
        "failures_output": str(args.failures_out),
    }
    return summary, failures


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="generate validated Whisper training sentences through a vLLM OpenAI API"
    )
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations.tsv"))
    parser.add_argument("--out", type=Path, required=True, help="atomic resumable JSONL output")
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--failures-out", type=Path, default=None)
    parser.add_argument("--sentences-per-term", type=int, default=8)
    parser.add_argument("--replay-ratio", type=float, default=0.25)
    parser.add_argument("--control-ratio", type=float, default=0.05)
    parser.add_argument("--noise-share", type=float, default=0.5)
    parser.add_argument("--control-duration-sec", type=float, default=1.0)
    parser.add_argument("--min-chars", type=int, default=18)
    parser.add_argument("--max-chars", type=int, default=80)
    parser.add_argument(
        "--domain-hint",
        default="",
        help="optional subject-area hint; empty keeps annotations domain-agnostic",
    )
    parser.add_argument(
        "--prompt-max-terms", type=int, default=200,
        help="maximum representative annotation terms included in one prompt",
    )
    parser.add_argument(
        "--prompt-max-chars", type=int, default=1200,
        help="maximum JSON character count for the prompt term subset",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--oversample", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-backoff-sec", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="resume validated IDs already present in --out (default: true)",
    )
    args = parser.parse_args(argv)
    if args.summary_out is None:
        args.summary_out = args.out.with_suffix(args.out.suffix + ".summary.json")
    if args.failures_out is None:
        args.failures_out = args.out.with_suffix(args.out.suffix + ".failures.jsonl")
    if args.sentences_per_term < 1:
        parser.error("--sentences-per-term must be positive")
    for name in ("replay_ratio", "control_ratio"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    if not 0 <= args.noise_share <= 1:
        parser.error("--noise-share must be between zero and one")
    if args.concurrency < 1 or args.oversample < 1:
        parser.error("--concurrency and --oversample must be positive")
    if args.prompt_max_terms < 1 or args.prompt_max_chars < 1:
        parser.error("prompt limits must be positive")
    if args.max_retries < 0 or args.retry_backoff_sec < 0:
        parser.error("retry values must be non-negative")
    if args.timeout <= 0 or args.max_tokens < 1:
        parser.error("--timeout and --max-tokens must be positive")
    if args.top_k < 1 or not 0 <= args.top_p <= 1:
        parser.error("--top-k must be positive and --top-p must be between zero and one")
    if not math.isfinite(args.presence_penalty):
        parser.error("--presence-penalty must be finite")
    SentenceLimits(args.min_chars, args.max_chars)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        summary, failures = asyncio.run(run_generation(args))
        atomic_write_json(args.summary_out, summary)
        atomic_write_jsonl(args.failures_out, failures)
    except (CorpusError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2
    LOGGER.info(
        "generation %s: completed=%d/%d, missing=%d; summary=%s",
        summary["status"],
        summary["completed"],
        summary["planned"],
        summary["missing"],
        args.summary_out,
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
