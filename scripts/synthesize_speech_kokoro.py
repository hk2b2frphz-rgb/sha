#!/usr/bin/env python3
"""発話例文 JSONL を Kokoro-82M (日本語) で音声合成する。

synthesize_speech.py (Qwen3-TTS) と同じ入出力形式:
  入力: tts_text (無ければ sentence) を読む JSONL
  出力: <out-dir>/wav/<id>.wav (モノラル16bit) と manifest.jsonl
        {"id","term","sentence","wav","duration_sec","voice"}

Kokoro は misaki[ja] と full unidic 辞書を必要とし、term2speech本体の
依存 (unidic-lite 等) と衝突するため、隔離env で動かす想定:
  uv run --isolated --no-project --with "kokoro>=0.9.4" --with "misaki[ja]" \
      --with unidic --with pyopenjtalk --with soundfile --with numpy \
      --with "torch==<ver>" --with "torchaudio==<ver>" \
      python scripts/synthesize_speech_kokoro.py --sentences ... --out-dir ...
辞書は事前に `python -m unidic download` しておくこと。
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
logger = logging.getLogger("synthesize_speech_kokoro")

SAMPLE_RATE = 24000  # Kokoro native output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="例文 JSONL を Kokoro-82M で音声合成する")
    parser.add_argument("--sentences", type=Path, required=True, help="例文 JSONL")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="hexgrad/Kokoro-82M")
    parser.add_argument("--lang-code", default="j", help="Kokoro lang_code (日本語は 'j')")
    # 日本語プリセット: jf_alpha, jf_gongitsune, jf_nezumi, jf_tebukuro (女性),
    # jm_kumo (男性)
    parser.add_argument("--voice", default="jf_alpha", help="Kokoro 日本語ボイス名")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--lead-silence-ms", type=int, default=0, help="WAV先頭に付与する無音（ミリ秒）")
    parser.add_argument("--device", default="cuda")
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


def load_pipeline(args: argparse.Namespace) -> Any:
    try:
        from kokoro import KPipeline
    except ImportError as exc:
        raise SystemExit(
            "kokoro をimportできません。隔離env (--with 'kokoro>=0.9.4' --with 'misaki[ja]') "
            "で実行してください。"
        ) from exc
    logger.info("Kokoro をロード中: %s (device=%s, lang=%s)", args.model_id, args.device, args.lang_code)
    pipeline = KPipeline(lang_code=args.lang_code, repo_id=args.model_id, device=args.device)
    logger.info("ロード完了")
    return pipeline


def synthesize(pipeline: Any, args: argparse.Namespace, text: str) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for result in pipeline(text, voice=args.voice, speed=args.speed):
        audio = result[2]
        if audio is None:
            continue
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))
    if not chunks:
        raise RuntimeError("Kokoro returned no audio chunks")
    return np.concatenate(chunks).astype(np.float32, copy=False)


def add_lead_silence(audio: np.ndarray, sample_rate: int, milliseconds: int) -> np.ndarray:
    """Add leading silence so playback/streaming does not clip the first phoneme."""
    samples = round(sample_rate * milliseconds / 1000)
    if samples <= 0:
        return audio
    return np.concatenate((np.zeros(samples, dtype=np.float32), audio.astype(np.float32, copy=False)))


def build_manifest_entry(
    rec: dict[str, Any], synthesis_text: str, wav: str, duration_sec: float, voice: str
) -> dict[str, Any]:
    """Preserve the kanji ASR target while recording what Kokoro actually read."""
    target = str(rec.get("sentence", rec.get("text", "")))
    entry = dict(rec)
    entry.update(
        {
            "id": rec["id"],
            "sentence": target,
            "tts_text": str(rec.get("tts_text", target)),
            "synthesis_text": synthesis_text,
            "wav": wav,
            "duration_sec": duration_sec,
            "voice": voice,
        }
    )
    return entry


def main() -> None:
    args = parse_args()
    records = load_sentences(args.sentences)
    logger.info("%d 文を音声合成します", len(records))

    pipeline = load_pipeline(args)

    wav_dir = args.out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.jsonl"

    import soundfile as sf

    start = time.monotonic()
    with manifest_path.open("w", encoding="utf-8") as fh:
        for i, rec in enumerate(records, 1):
            text = rec.get("tts_text") or rec["sentence"]
            audio = synthesize(pipeline, args, text)
            audio = add_lead_silence(audio, SAMPLE_RATE, args.lead_silence_ms)
            wav_path = wav_dir / f"{rec['id']}.wav"
            sf.write(wav_path, audio, SAMPLE_RATE, subtype="PCM_16")
            # Keep the ASR target separate from the pronunciation-adjusted text.
            # Evaluation reads ``sentence`` as its reference, so writing ``text``
            # (which is normally tts_text) here would incorrectly score hiragana
            # instead of the original technical-term spelling.
            entry = build_manifest_entry(
                rec,
                text,
                str(wav_path.relative_to(args.out_dir)),
                round(audio.size / SAMPLE_RATE, 2),
                args.voice,
            )
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
