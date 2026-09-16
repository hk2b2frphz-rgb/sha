#!/usr/bin/env python3
"""Join a generated corpus and sharded TTS manifests for Whisper training.

The generated corpus is the source of truth for the ASR label.  In particular,
``sentence`` (the original kanji spelling) must never be replaced by
``tts_text`` (the reading sent to the synthesizer).  The output schema is one
JSON object per line with at least::

    {"id": "...", "audio": "/abs/file.wav", "text": "..."}

Additional provenance fields are retained so that the train/dev splitter can
stratify technical-term, replay, and no-speech control examples.
"""
from __future__ import annotations

import argparse
import json
import os
import wave
from pathlib import Path
from typing import Any


PROVENANCE_KEYS = (
    "kind",
    "term",
    "reading",
    "source_term",
    "source_index",
    "tts_text",
    "synthesis_text",
    "pronunciation_check",
    "pronunciation_asr_text",
    "pronunciation_match",
    "pronunciation_check_reason",
    "pronunciation_expected_reading",
    "pronunciation_observed_reading",
    "pronunciation_mora_distance",
    "pronunciation_content_edits",
    "reading_fallback_used",
    "speaker",
    "control_duration_sec",
    "control_seed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TTS結果をWhisper学習manifestへ変換")
    parser.add_argument("--tts-input", type=Path, required=True, help="生成済みcorpus JSONL")
    parser.add_argument(
        "--synth-dir",
        type=Path,
        action="append",
        required=True,
        help="synthesize_speech.py の --out-dir。シャード分だけ複数指定",
    )
    parser.add_argument("--out", type=Path, required=True, help="学習manifest JSONL")
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--min-duration", type=float, default=0.05)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="欠損ID/WAVを除外して部分manifestを許可する（既定はfail-fast）",
    )
    return parser.parse_args()


def load_jsonl(path: Path, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(rec, dict) or not str(rec.get("id", "")).strip():
            raise SystemExit(f"{path}:{line_no}: object with a non-empty id is required")
        rec_id = str(rec["id"])
        if rec_id in seen:
            raise SystemExit(f"{path}:{line_no}: duplicate id: {rec_id}")
        seen.add(rec_id)
        rows.append(rec)
    if not rows and not allow_empty:
        raise SystemExit(f"JSONL is empty: {path}")
    return rows


def wav_duration(path: Path) -> float:
    """Read a PCM WAV duration without importing the TTS environment."""
    try:
        with wave.open(str(path), "rb") as fh:
            rate = fh.getframerate()
            return fh.getnframes() / float(rate) if rate else 0.0
    except (wave.Error, OSError) as exc:
        raise ValueError(f"cannot read WAV: {path}: {exc}") from exc


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def build_rows(
    targets: list[dict[str, Any]],
    synth_dirs: list[Path],
    *,
    min_duration: float,
    max_duration: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_by_id = {str(row["id"]): row for row in targets}
    synthesized: dict[str, tuple[dict[str, Any], Path]] = {}
    extras: list[str] = []

    for synth_dir in synth_dirs:
        manifest = synth_dir / "manifest.jsonl"
        if not manifest.is_file():
            raise SystemExit(f"合成manifestが見つかりません: {manifest}")
        for rec in load_jsonl(manifest, allow_empty=True):
            rec_id = str(rec["id"])
            if rec_id not in target_by_id:
                extras.append(rec_id)
                continue
            if rec_id in synthesized:
                previous_dir = synthesized[rec_id][1]
                raise SystemExit(
                    f"duplicate synthesized id {rec_id}: {previous_dir} and {synth_dir}"
                )
            synthesized[rec_id] = (rec, synth_dir)

    output: list[dict[str, Any]] = []
    missing_ids: list[str] = []
    missing_wavs: list[str] = []
    invalid_audio: list[str] = []
    for target in targets:  # retain corpus order, independent of shard completion order
        rec_id = str(target["id"])
        pair = synthesized.get(rec_id)
        if pair is None:
            missing_ids.append(rec_id)
            continue
        synth, synth_dir = pair
        wav_value = str(synth.get("wav", "")).strip()
        if not wav_value:
            missing_wavs.append(rec_id)
            continue
        wav = Path(wav_value)
        wav_path = wav if wav.is_absolute() else synth_dir / wav
        if not wav_path.is_file():
            missing_wavs.append(rec_id)
            continue
        try:
            measured_duration = wav_duration(wav_path)
        except ValueError:
            invalid_audio.append(rec_id)
            continue
        duration = float(synth.get("duration_sec") or measured_duration)
        # Trust neither a stale manifest nor an obviously truncated file.
        if (
            measured_duration < min_duration
            or measured_duration > max_duration
            or abs(duration - measured_duration) > max(0.25, measured_duration * 0.05)
        ):
            invalid_audio.append(rec_id)
            continue

        if "sentence" not in target:
            raise SystemExit(f"corpus record {rec_id} has no sentence field")
        row: dict[str, Any] = {
            "id": rec_id,
            "audio": str(wav_path.resolve()),
            # Empty text is intentional for silence/noise controls.
            "text": str(target.get("sentence", "")),
            "duration_sec": round(measured_duration, 3),
        }
        for key in PROVENANCE_KEYS:
            if key in target:
                row[key] = target[key]
            elif key in synth:
                row[key] = synth[key]
        output.append(row)

    summary = {
        "expected": len(targets),
        "written": len(output),
        "missing_ids": missing_ids,
        "missing_wavs": missing_wavs,
        "invalid_audio": invalid_audio,
        "extra_synthesized_ids": sorted(set(extras)),
        "kind_counts": {},
    }
    for row in output:
        kind = str(row.get("kind", "term"))
        summary["kind_counts"][kind] = summary["kind_counts"].get(kind, 0) + 1
    return output, summary


def main() -> None:
    args = parse_args()
    if args.min_duration <= 0 or args.max_duration <= args.min_duration:
        raise SystemExit("duration bounds must satisfy 0 < min < max")
    targets = load_jsonl(args.tts_input)
    rows, summary = build_rows(
        targets,
        args.synth_dir,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )
    incomplete = any(
        summary[key] for key in ("missing_ids", "missing_wavs", "invalid_audio")
    )
    if incomplete and not args.allow_missing:
        raise SystemExit(
            "TTS corpus is incomplete: "
            f"missing_ids={len(summary['missing_ids'])}, "
            f"missing_wavs={len(summary['missing_wavs'])}, "
            f"invalid_audio={len(summary['invalid_audio'])}. "
            "See shard manifests/logs; use --allow-missing only for diagnostics."
        )
    if not rows:
        raise SystemExit("学習manifestが空になりました")

    atomic_write_jsonl(args.out, rows)
    summary_path = args.summary or args.out.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {len(rows)}/{len(targets)} records -> {args.out} "
        f"kinds={summary['kind_counts']}"
    )


if __name__ == "__main__":
    main()
