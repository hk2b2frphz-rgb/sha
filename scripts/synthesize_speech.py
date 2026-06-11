#!/usr/bin/env python3
"""
発話例文 JSONL を Qwen3-TTS で音声合成する。

入力:  generate_sentences.py の出力 JSONL (--sentences)
出力:  --out-dir 配下に
       wav/<id>.wav        モノラル 16bit WAV
       manifest.jsonl      {"id", "term", "sentence", "wav", "duration_sec", "speaker"}

使い方:
  uv run python scripts/synthesize_speech.py \
      --sentences out/sentences.jsonl --out-dir out/audio
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("synthesize_speech")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="例文 JSONL を Qwen3-TTS で音声合成する")
    parser.add_argument("--sentences", type=Path, required=True, help="例文 JSONL")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    # プリセット話者: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee
    # 日本語は Ono_Anna が自然 (Vivian 等は中国語系でアクセントが不自然になりがち)
    parser.add_argument("--speaker", default="Ono_Anna", help="Qwen3-TTS プリセット話者名")
    parser.add_argument("--language", default="Japanese")
    parser.add_argument("--instruct", default=None, help="話し方のスタイル指示 (省略可)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    return parser.parse_args()


def load_sentences(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    if not records:
        raise SystemExit(f"例文が見つかりません: {path}")
    return records


def load_model(args: argparse.Namespace) -> Any:
    import torch
    from qwen_tts import Qwen3TTSModel

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    logger.info("Qwen3-TTS をロード中: %s (device=%s)", args.model, args.device)
    model = Qwen3TTSModel.from_pretrained(args.model, device_map=args.device, dtype=dtype)
    logger.info("ロード完了")
    return model


def synthesize(model: Any, args: argparse.Namespace, text: str) -> tuple[np.ndarray, int]:
    import torch

    kwargs: dict[str, Any] = {
        "text": text,
        "language": args.language,
        "speaker": args.speaker,
    }
    if args.instruct:
        kwargs["instruct"] = args.instruct
    with torch.no_grad():
        wavs, sr = model.generate_custom_voice(**kwargs)
    audio = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
    if hasattr(audio, "cpu"):
        audio = audio.cpu().numpy()
    return np.asarray(audio, dtype=np.float32).squeeze(), int(sr)


def main() -> None:
    args = parse_args()
    records = load_sentences(args.sentences)
    logger.info("%d 文を音声合成します", len(records))

    model = load_model(args)

    wav_dir = args.out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.jsonl"

    import soundfile as sf

    start = time.monotonic()
    with manifest_path.open("w", encoding="utf-8") as fh:
        for i, rec in enumerate(records, 1):
            text = rec["sentence"]
            audio, sr = synthesize(model, args, text)
            wav_path = wav_dir / f"{rec['id']}.wav"
            sf.write(wav_path, audio, sr, subtype="PCM_16")
            entry = {
                "id": rec["id"],
                "term": rec.get("term", ""),
                "sentence": text,
                "wav": str(wav_path.relative_to(args.out_dir)),
                "duration_sec": round(audio.size / sr, 2),
                "speaker": args.speaker,
            }
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fh.flush()  # 中断してもここまでの manifest は残る
            elapsed = time.monotonic() - start
            eta = elapsed / i * (len(records) - i)
            logger.info("(%d/%d) %s [%.1fs] %s | 経過 %.0f 秒 / 残り目安 %.0f 秒",
                        i, len(records), wav_path.name,
                        entry["duration_sec"], text[:30], elapsed, eta)

    logger.info("完了: %s", manifest_path)


if __name__ == "__main__":
    main()
