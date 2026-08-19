#!/usr/bin/env python3
"""Generate a tiny dummy dataset so the whole loop can be exercised on the HPC.

The audio is synthetic (formant-ish tones, not speech), so the recognizer will
produce nonsense and CER will sit near 1.0. That is expected and fine: the smoke
test checks that every stage runs, writes its files and produces numbers, not
that the numbers are good.

  uv run python scripts/make_smoke_data.py --out-dir out/smoke --train 12 --test 8

Writes:
  <out-dir>/train_wav/train_XX.wav  training audio, kept apart from the test set
  <out-dir>/wav/test_XX.wav        the test wav dir, holds nothing else
  <out-dir>/train_manifest.jsonl   {"id","audio","text","term"}
  <out-dir>/refs.txt               reference text, one per test wav (natural order)

Train and test audio live in separate directories on purpose: build_test_manifest.py
pairs positional references with every wav it finds, so the test directory must
contain exactly the wavs those references belong to.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000

# Short domain-ish sentences; the terms let the loop exercise term recall and
# the prompt_terms knob without needing the real corpus.
SENTENCES: list[tuple[str, str]] = [
    ("深層学習", "深層学習の研究発表を聴講した。"),
    ("音声認識", "音声認識の精度を評価する。"),
    ("転移学習", "転移学習で少量データに適応する。"),
    ("正規化層", "正規化層を挿入して学習を安定させる。"),
    ("誤差逆伝播", "誤差逆伝播で勾配を計算する。"),
    ("量子化", "量子化によって推論を高速化する。"),
    ("符号化器", "符号化器の出力を復号器に渡す。"),
    ("注意機構", "注意機構が長距離依存を捉える。"),
    ("語彙集合", "語彙集合の大きさを制限する。"),
    ("推論時間", "推論時間を実測して比較する。"),
    ("学習率", "学習率の減衰を調整する。"),
    ("評価指標", "評価指標には文字誤り率を用いる。"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="tiny synthetic dataset for the smoke test")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train", type=int, default=12, help="training utterances")
    parser.add_argument("--test", type=int, default=8, help="test utterances")
    parser.add_argument("--seconds", type=float, default=1.5, help="duration per utterance")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def make_wave(seconds: float, seed: int) -> np.ndarray:
    """A short, band-limited, speech-shaped-ish tone burst."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    base = rng.uniform(110, 220)
    audio = np.zeros_like(t)
    for harmonic, gain in ((1, 1.0), (2, 0.5), (3, 0.25), (5, 0.1)):
        audio += gain * np.sin(2 * np.pi * base * harmonic * t + rng.uniform(0, 6.28))
    # Amplitude envelope so it is not a constant drone.
    envelope = 0.5 * (1 - np.cos(2 * np.pi * np.clip(t / seconds, 0, 1) * 3)) / 2 + 0.2
    audio = audio * envelope
    audio += rng.normal(0, 0.01, size=audio.shape)
    peak = float(np.max(np.abs(audio))) or 1.0
    return (0.7 * audio / peak).astype("float32")


def main() -> None:
    args = parse_args()
    wav_dir = args.out_dir / "wav"
    train_wav_dir = args.out_dir / "train_wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    train_wav_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.out_dir / "train_manifest.jsonl"
    with train_path.open("w", encoding="utf-8") as fh:
        for i in range(args.train):
            term, sentence = SENTENCES[i % len(SENTENCES)]
            wav_path = (train_wav_dir / f"train_{i:03d}.wav").resolve()
            sf.write(wav_path, make_wave(args.seconds, args.seed + i), SAMPLE_RATE, subtype="PCM_16")
            fh.write(
                json.dumps(
                    {
                        "id": f"train_{i:03d}",
                        "audio": str(wav_path),
                        "text": sentence,
                        "term": term,
                        "duration_sec": args.seconds,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    refs = []
    for i in range(args.test):
        term, sentence = SENTENCES[(i + 5) % len(SENTENCES)]
        wav_path = wav_dir / f"test_{i:03d}.wav"
        sf.write(wav_path, make_wave(args.seconds, args.seed + 1000 + i), SAMPLE_RATE, subtype="PCM_16")
        refs.append(sentence)
    (args.out_dir / "refs.txt").write_text("\n".join(refs) + "\n", encoding="utf-8")

    print(f"[smoke-data] train manifest: {train_path} ({args.train} rows)")
    print(f"[smoke-data] train wavs:     {train_wav_dir} ({args.train} files)")
    print(f"[smoke-data] test wavs:      {wav_dir} ({args.test} files)")
    print(f"[smoke-data] references:     {args.out_dir / 'refs.txt'}")
    print("[smoke-data] NOTE: the audio is synthetic, so CER near 1.0 is expected.")


if __name__ == "__main__":
    main()
