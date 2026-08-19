#!/usr/bin/env python3
"""Shared test wavs -> whisper-streaming evaluation manifest.

evaluate_whisper_streaming.py reads {"id","wav","sentence"} per line and uses
`sentence` as the reference for CER/WER. This builds that manifest from a wav
directory plus an optional references file.

References file (--references) supports two formats, auto-detected:
  1. Positional (no keys): one reference text per line, in the same order as
     the wavs. Line i is paired with the i-th wav (natural-sorted). Blank lines
     and lines starting with '#' are ignored.
  2. Keyed TSV: each line is "<key>\t<text>" where key is the wav filename
     (with or without extension) or stem.
If a line contains a TAB, keyed mode is used; otherwise positional mode.
Wavs without a reference get sentence="" (transcribe-only; metrics meaningless).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="build eval manifest from test wavs")
    parser.add_argument("--wav-dir", type=Path, required=True, help="wav directory (recursive)")
    parser.add_argument("--references", type=Path, default=None, help="reference texts (positional or keyed TSV)")
    parser.add_argument("--out", type=Path, required=True, help="output manifest.jsonl")
    return parser.parse_args()


def natural_key(path: Path) -> list:
    # Natural sort so 2.wav < 10.wav (matches human "in order").
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def read_ref_lines(path: Path) -> list[str]:
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        lines.append(line)
    return lines


def main() -> None:
    args = parse_args()
    wavs = sorted((p for p in args.wav_dir.rglob("*.wav") if p.is_file()), key=natural_key)
    if not wavs:
        raise SystemExit(f"no wav found under: {args.wav_dir}")

    refs_by_key: dict[str, str] = {}
    refs_positional: list[str] = []
    if args.references is not None:
        ref_lines = read_ref_lines(args.references)
        keyed = any("\t" in line for line in ref_lines)
        if keyed:
            for line in ref_lines:
                cols = line.split("\t")
                if len(cols) < 2:
                    continue
                key, text = cols[0].strip(), cols[1].strip()
                if key.lower() in ("id", "key", "filename", "wav"):  # header
                    continue
                refs_by_key[key] = text
                refs_by_key[Path(key).name] = text
                refs_by_key[Path(key).stem] = text
        else:
            refs_positional = [line.strip() for line in ref_lines]
            if len(refs_positional) != len(wavs):
                print(
                    f"[WARN] reference count ({len(refs_positional)}) != wav count "
                    f"({len(wavs)}); pairing by position up to the shorter length."
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    matched = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for i, wav in enumerate(wavs):
            if refs_by_key:
                ref = refs_by_key.get(wav.name) or refs_by_key.get(wav.stem) or ""
            elif refs_positional:
                ref = refs_positional[i] if i < len(refs_positional) else ""
            else:
                ref = ""
            if ref:
                matched += 1
            fh.write(
                json.dumps(
                    {"id": wav.stem, "wav": str(wav.resolve()), "sentence": ref},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote {len(wavs)} records -> {args.out} (with_reference={matched})")
    if args.references is not None and matched == 0:
        print("[WARN] references given but 0 matched. Check order/keys.")


if __name__ == "__main__":
    main()
