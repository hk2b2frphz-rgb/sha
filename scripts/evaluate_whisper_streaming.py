#!/usr/bin/env python3
"""Evaluate CTranslate2 Whisper with ufal/whisper_streaming online policy."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.asr_text import (  # noqa: E402  (needs the sys.path bootstrap above)
    JA_PUNCT_RE,
    JapaneseTokenizer,
    edit_distance,
    error_rate,
    normalize_text,
)

__all__ = ["JA_PUNCT_RE", "JapaneseTokenizer", "edit_distance", "error_rate", "normalize_text"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="whisper-streamingでASR評価を行う")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"config must be a mapping: {path}")
    return data


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def import_whisper_online(repo_dir: Path) -> Any:
    module_path = repo_dir / "whisper_online.py"
    if not module_path.exists():
        raise SystemExit(
            f"whisper_online.py が見つかりません: {module_path}\n"
            "PBSでは vendor/whisper_streaming を自動cloneします。手動実行時はrepo_dirを確認してください。"
        )
    sys.path.insert(0, str(repo_dir))
    spec = importlib.util.spec_from_file_location("whisper_online", module_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class ConfigurableFasterWhisperASR:
    """Small adapter for whisper_streaming.OnlineASRProcessor."""

    model_dir: str
    language: str | None
    task: str
    device: str
    device_index: int
    compute_type: str
    beam_size: int
    vad_filter: bool

    sep: str = ""

    def __post_init__(self) -> None:
        from faster_whisper import WhisperModel

        self.original_language = None if self.language == "auto" else self.language
        self.model = WhisperModel(
            self.model_dir,
            device=self.device,
            device_index=self.device_index,
            compute_type=self.compute_type,
        )

    def transcribe(self, audio: Any, init_prompt: str = "") -> list[Any]:
        segments, _info = self.model.transcribe(
            audio,
            language=self.original_language,
            task=self.task,
            initial_prompt=init_prompt,
            beam_size=self.beam_size,
            word_timestamps=True,
            condition_on_previous_text=True,
            vad_filter=self.vad_filter,
        )
        return list(segments)

    def ts_words(self, segments: list[Any]) -> list[tuple[float, float, str]]:
        out: list[tuple[float, float, str]] = []
        for segment in segments:
            if getattr(segment, "no_speech_prob", 0.0) > 0.9:
                continue
            for word in segment.words or []:
                out.append((float(word.start), float(word.end), str(word.word)))
        return out

    def segments_end_ts(self, segments: list[Any]) -> list[float]:
        return [float(s.end) for s in segments]


def load_manifest(path: Path, audio_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        rec = json.loads(raw)
        wav = Path(str(rec["wav"]))
        wav_path = wav if wav.is_absolute() else audio_dir / wav
        if not wav_path.exists():
            raise SystemExit(f"{path}:{line_no}: wav not found: {wav_path}")
        rec["wav_path"] = str(wav_path)
        records.append(rec)
    if not records:
        raise SystemExit(f"manifest is empty: {path}")
    return records


def run_streaming(module: Any, online: Any, wav_path: str, min_chunk_size: float) -> tuple[str, float, float]:
    audio = module.load_audio(wav_path)
    duration = len(audio) / float(module.OnlineASRProcessor.SAMPLING_RATE)
    online.init()
    pieces: list[str] = []
    beg = 0.0
    start = time.monotonic()
    while beg < duration:
        end = min(duration, beg + min_chunk_size)
        chunk = module.load_audio_chunk(wav_path, beg, end)
        online.insert_audio_chunk(chunk)
        out = online.process_iter()
        if out[2]:
            pieces.append(out[2])
        beg = end
    final = online.finish()
    if final[2]:
        pieces.append(final[2])
    elapsed = time.monotonic() - start
    return "".join(pieces).strip(), duration, elapsed


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    cfg = load_config(args.config)

    dataset_cfg = cfg.get("dataset", {})
    output_cfg = cfg.get("output", {})
    ws_cfg = cfg.get("whisper_streaming", {})
    model_cfg = cfg.get("model", {})
    metrics_cfg = cfg.get("metrics", {})

    manifest = args.manifest or resolve_path(dataset_cfg.get("manifest", "out/whisper_streaming_eval/audio/manifest.jsonl"), project_root)
    audio_dir = resolve_path(dataset_cfg.get("audio_dir", manifest.parent), project_root)
    out_dir = args.out_dir or resolve_path(output_cfg.get("dir", "experiments/whisper_streaming_eval"), project_root)
    repo_dir_value = os.environ.get("WHISPER_STREAMING_DIR") or ws_cfg.get("repo_dir", "vendor/whisper_streaming")
    repo_dir = resolve_path(repo_dir_value, project_root)
    model_dir = args.model_dir or str(model_cfg.get("model_dir", ""))
    if not model_dir or model_dir == "/path/to/your/ctranslate2-finetuned-whisper":
        raise SystemExit("config の model.model_dir にCTranslate2モデルディレクトリを指定してください")
    model_dir_path = Path(model_dir)
    if not model_dir_path.is_absolute() and (project_root / model_dir_path).exists():
        model_dir = str(project_root / model_dir_path)

    module = import_whisper_online(repo_dir)
    records = load_manifest(manifest, audio_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    asr = ConfigurableFasterWhisperASR(
        model_dir=model_dir,
        language=str(ws_cfg.get("language", "ja")),
        task=str(ws_cfg.get("task", "transcribe")),
        device=str(model_cfg.get("device", "cuda")),
        device_index=int(model_cfg.get("device_index", 0)),
        compute_type=str(model_cfg.get("compute_type", "float16")),
        beam_size=int(model_cfg.get("beam_size", 5)),
        vad_filter=bool(ws_cfg.get("vad", False)),
    )
    online = module.OnlineASRProcessor(
        asr,
        tokenizer=None,
        buffer_trimming=(str(ws_cfg.get("buffer_trimming", "segment")), float(ws_cfg.get("buffer_trimming_sec", 15))),
        logfile=sys.stderr,
    )
    min_chunk_size = float(ws_cfg.get("min_chunk_size", 1.0))
    if bool(ws_cfg.get("warmup", True)):
        warm = module.load_audio(records[0]["wav_path"])[: module.OnlineASRProcessor.SAMPLING_RATE]
        if len(warm):
            asr.transcribe(warm)

    tokenizer = JapaneseTokenizer(bool(metrics_cfg.get("japanese_word_tokenizer", True)))
    predictions_path = out_dir / str(output_cfg.get("predictions_jsonl", "predictions.jsonl"))
    results: list[dict[str, Any]] = []
    total_audio = 0.0
    total_wall = 0.0
    with predictions_path.open("w", encoding="utf-8") as fh:
        for idx, rec in enumerate(records, 1):
            hyp, duration, wall = run_streaming(module, online, rec["wav_path"], min_chunk_size)
            ref = str(rec.get("sentence") or rec.get("text") or rec.get("tts_text") or "")
            ref_chars = list(normalize_text(ref))
            hyp_chars = list(normalize_text(hyp))
            ref_words = tokenizer.words(ref)
            hyp_words = tokenizer.words(hyp)
            row = {
                "id": rec.get("id", f"{idx:04d}"),
                "wav": rec.get("wav"),
                "reference": ref,
                "prediction": hyp,
                "duration_sec": duration,
                "wall_sec": wall,
                "rtf": wall / duration if duration else None,
                "speed_x": duration / wall if wall else None,
                "cer": error_rate(ref_chars, hyp_chars),
                "wer": error_rate(ref_words, hyp_words),
            }
            total_audio += duration
            total_wall += wall
            results.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(
                f"[{idx}/{len(records)}] {row['id']} CER={row['cer']:.3f} "
                f"WER={row['wer']:.3f} speed={row['speed_x']:.2f}x"
            )

    n = len(results)
    summary = {
        "n": n,
        "manifest": str(manifest),
        "model_dir": model_dir,
        "tokenizer": tokenizer.mode,
        "cer": sum(r["cer"] for r in results) / n,
        "wer": sum(r["wer"] for r in results) / n,
        "total_audio_sec": total_audio,
        "total_wall_sec": total_wall,
        "rtf": total_wall / total_audio if total_audio else None,
        "speed_x": total_audio / total_wall if total_wall else None,
        "min_chunk_size": min_chunk_size,
    }
    summary_path = out_dir / str(output_cfg.get("summary_json", "summary.json"))
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report_path = out_dir / str(output_cfg.get("report_md", "report.md"))
    report_path.write_text(
        "\n".join(
            [
                "# whisper-streaming evaluation",
                "",
                f"- n: {summary['n']}",
                f"- model_dir: `{summary['model_dir']}`",
                f"- CER: {summary['cer']:.4f}",
                f"- WER: {summary['wer']:.4f} ({summary['tokenizer']})",
                f"- speed: {summary['speed_x']:.2f}x audio / RTF {summary['rtf']:.4f}",
                f"- total audio: {summary['total_audio_sec']:.2f} sec",
                f"- total wall: {summary['total_wall_sec']:.2f} sec",
                f"- predictions: `{predictions_path}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
