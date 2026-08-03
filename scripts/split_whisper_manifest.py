#!/usr/bin/env python3
"""Whisper manifest を再現可能な stratified train/dev に分割する。

``kind / source_term / term`` の組を層として扱う。2件以上ある層は train と
dev の双方に最低1件を置き、さらに同じ専門用語の正例が2件以上あれば、層を
またぐ場合も可能な限り双方へ配置する。audio/wav または id の重複は出力前に
除去し、内容が食い違う衝突はエラーにする。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


StratumKey = tuple[str, str, str]


@dataclass(frozen=True)
class SplitResult:
    train: list[dict[str, Any]]
    dev: list[dict[str, Any]]
    duplicates_removed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="split Whisper manifest into train/dev")
    parser.add_argument("--manifest", "--input", dest="manifest", type=Path, required=True)
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--dev-out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def canonical_record(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def record_identity_keys(record: dict[str, Any]) -> tuple[str, ...]:
    """データ漏洩を防ぐため、音声パスとIDをそれぞれ一意キーとして使う。"""
    keys: list[str] = []
    audio = record.get("audio") or record.get("wav")
    if audio is not None and str(audio).strip():
        normalized_audio = os.path.normcase(os.path.normpath(str(audio).strip()))
        keys.append(f"audio:{normalized_audio}")
    rec_id = record.get("id")
    if rec_id is not None and str(rec_id).strip():
        keys.append(f"id:{str(rec_id).strip()}")
    if not keys:
        keys.append("record:" + hashlib.sha256(canonical_record(record).encode("utf-8")).hexdigest())
    return tuple(keys)


def deduplicate_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """完全重複を除去し、同一ID/音声で内容が違う衝突は拒否する。"""
    unique: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    duplicate_count = 0
    for index, row in enumerate(rows, 1):
        canonical = canonical_record(row)
        keys = record_identity_keys(row)
        collisions = [(key, seen[key]) for key in keys if key in seen]
        if collisions:
            if all(previous == canonical for _key, previous in collisions):
                duplicate_count += 1
                continue
            collision_keys = ", ".join(key for key, _previous in collisions)
            raise ValueError(
                f"manifest row {index} conflicts with an earlier row ({collision_keys})"
            )
        unique.append(dict(row))
        for key in keys:
            seen[key] = canonical
    return unique, duplicate_count


def stratification_key(record: dict[str, Any]) -> StratumKey:
    return (
        str(record.get("kind") or "").strip(),
        str(record.get("source_term") or "").strip(),
        str(record.get("term") or "").strip(),
    )


def positive_term(record: dict[str, Any]) -> str | None:
    """正例の専門用語を返す。明示的な負例や本文に用語が無い行は除外する。"""
    term = str(record.get("term") or record.get("source_term") or "").strip()
    if not term:
        return None
    kind = str(record.get("kind") or "").strip().casefold().replace("-", "_")
    negative_kinds = {
        "negative",
        "neg",
        "hard_negative",
        "distractor",
        "control",
        "negative_example",
        "replay",
        "silence",
        "no_speech",
        "noise",
        "blank",
        "負例",
        "陰性",
    }
    if (
        kind in negative_kinds
        or kind.endswith("_negative")
        or kind.startswith("negative_")
    ):
        return None
    text = str(record.get("text") or record.get("sentence") or "").strip()
    if text:
        normalized_term = unicodedata.normalize("NFKC", term)
        normalized_text = unicodedata.normalize("NFKC", text)
        if normalized_term not in normalized_text:
            return None
    return term


def stable_score(seed: int, namespace: str, value: str) -> int:
    payload = f"{seed}\0{namespace}\0{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def split_rows(rows: list[dict[str, Any]], *, dev_ratio: float, seed: int) -> SplitResult:
    if not 0.0 < dev_ratio < 1.0:
        raise ValueError("dev_ratio must be between 0 and 1")
    unique, duplicates_removed = deduplicate_rows(rows)
    if len(unique) < 2:
        raise ValueError("at least two unique manifest records are required")

    strata: dict[StratumKey, list[int]] = {}
    for index, row in enumerate(unique):
        strata.setdefault(stratification_key(row), []).append(index)

    dev_indexes: set[int] = set()
    for key in sorted(strata):
        indexes = sorted(
            strata[key],
            key=lambda index: (
                stable_score(seed, "stratum:" + "\0".join(key), canonical_record(unique[index])),
                canonical_record(unique[index]),
            ),
        )
        count = len(indexes)
        if count >= 2:
            dev_count = min(count - 1, max(1, int(count * dev_ratio + 0.5)))
        else:
            cutoff = int(dev_ratio * (1 << 256))
            dev_count = int(
                stable_score(seed, "singleton:" + "\0".join(key), canonical_record(unique[indexes[0]]))
                < cutoff
            )
        dev_indexes.update(indexes[:dev_count])

    # 同じ用語の正例が複数ある場合は、層をまたいでいても train/dev の双方へ置く。
    by_positive_term: dict[str, list[int]] = {}
    for index, row in enumerate(unique):
        term = positive_term(row)
        if term:
            by_positive_term.setdefault(term, []).append(index)
    for term in sorted(by_positive_term):
        indexes = by_positive_term[term]
        if len(indexes) < 2:
            continue
        in_dev = [index for index in indexes if index in dev_indexes]
        in_train = [index for index in indexes if index not in dev_indexes]
        ordered = sorted(
            indexes,
            key=lambda index: (
                stable_score(seed, "positive:" + term, canonical_record(unique[index])),
                canonical_record(unique[index]),
            ),
        )
        if not in_dev:
            dev_indexes.add(ordered[0])
        elif not in_train:
            dev_indexes.remove(ordered[0])

    # singleton層だけの小規模manifestでも両方を空にしない。
    if not dev_indexes:
        dev_indexes.add(
            min(
                range(len(unique)),
                key=lambda index: stable_score(seed, "nonempty-dev", canonical_record(unique[index])),
            )
        )
    if len(dev_indexes) == len(unique):
        dev_indexes.remove(
            min(
                dev_indexes,
                key=lambda index: stable_score(seed, "nonempty-train", canonical_record(unique[index])),
            )
        )

    train = [row for index, row in enumerate(unique) if index not in dev_indexes]
    dev = [row for index, row in enumerate(unique) if index in dev_indexes]
    train_keys = {key for row in train for key in record_identity_keys(row)}
    dev_keys = {key for row in dev for key in record_identity_keys(row)}
    overlap = train_keys & dev_keys
    if overlap:
        raise AssertionError(f"train/dev identity overlap: {sorted(overlap)[:3]}")
    return SplitResult(train=train, dev=dev, duplicates_removed=duplicates_removed)


def build_summary(
    result: SplitResult,
    *,
    input_count: int,
    dev_ratio: float,
    seed: int,
) -> dict[str, Any]:
    all_rows = result.train + result.dev
    strata: dict[StratumKey, dict[str, int]] = {}
    for split_name, rows in (("train", result.train), ("dev", result.dev)):
        for row in rows:
            key = stratification_key(row)
            counts = strata.setdefault(key, {"train": 0, "dev": 0})
            counts[split_name] += 1

    positive_counts: dict[str, dict[str, int]] = {}
    for split_name, rows in (("train", result.train), ("dev", result.dev)):
        for row in rows:
            term = positive_term(row)
            if not term:
                continue
            counts = positive_counts.setdefault(term, {"train": 0, "dev": 0})
            counts[split_name] += 1
    possible_both = sum(1 for counts in positive_counts.values() if sum(counts.values()) >= 2)
    present_both = sum(
        1
        for counts in positive_counts.values()
        if sum(counts.values()) >= 2 and counts["train"] > 0 and counts["dev"] > 0
    )

    return {
        "input_records": input_count,
        "unique_records": len(all_rows),
        "duplicates_removed": result.duplicates_removed,
        "train_records": len(result.train),
        "dev_records": len(result.dev),
        "requested_dev_ratio": dev_ratio,
        "actual_dev_ratio": len(result.dev) / len(all_rows),
        "seed": seed,
        "identity_overlap": 0,
        "strata_count": len(strata),
        "strata": [
            {
                "kind": key[0],
                "source_term": key[1],
                "term": key[2],
                "total": counts["train"] + counts["dev"],
                **counts,
            }
            for key, counts in sorted(strata.items())
        ],
        "positive_terms": len(positive_counts),
        "positive_terms_possible_in_both": possible_both,
        "positive_terms_present_in_both": present_both,
        "positive_term_counts": [
            {"term": term, "total": sum(counts.values()), **counts}
            for term, counts in sorted(positive_counts.items())
        ],
    }


def jsonl_text(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def atomic_write_many(payloads: list[tuple[Path, str]]) -> None:
    """全内容を一旦同一filesystem上へ書き切ってから各出力を atomic replace する。"""
    staged: list[tuple[Path, Path]] = []
    try:
        for destination, content in payloads:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
                staged.append((Path(handle.name), destination))
        for temporary, destination in staged:
            os.replace(temporary, destination)
    finally:
        for temporary, _destination in staged:
            temporary.unlink(missing_ok=True)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        record = json.loads(raw)
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
        rows.append(record)
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    return rows


def main() -> None:
    args = parse_args()
    summary_path = args.summary or args.train_out.with_name("split_summary.json")
    destinations = [args.train_out.absolute(), args.dev_out.absolute(), summary_path.absolute()]
    if len(set(destinations)) != len(destinations):
        raise SystemExit("--train-out, --dev-out, and --summary must be different paths")
    if args.manifest.absolute() in destinations:
        raise SystemExit("output paths must not overwrite the input manifest")
    try:
        rows = load_rows(args.manifest)
        result = split_rows(rows, dev_ratio=args.dev_ratio, seed=args.seed)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    summary = build_summary(
        result,
        input_count=len(rows),
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )
    summary.update(
        {
            "manifest": str(args.manifest),
            "train_manifest": str(args.train_out),
            "dev_manifest": str(args.dev_out),
        }
    )
    atomic_write_many(
        [
            (args.train_out, jsonl_text(result.train)),
            (args.dev_out, jsonl_text(result.dev)),
            (summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n"),
        ]
    )
    print(
        f"split {summary['unique_records']} unique records -> "
        f"train={summary['train_records']}, dev={summary['dev_records']} "
        f"(duplicates_removed={summary['duplicates_removed']}, "
        f"positive_terms_in_both={summary['positive_terms_present_in_both']}/"
        f"{summary['positive_terms_possible_in_both']})"
    )
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
