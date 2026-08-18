#!/usr/bin/env python3
"""Evaluate CTranslate2 Whisper with ufal/whisper_streaming online policy."""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.asr_text import (  # noqa: E402  (needs the sys.path bootstrap above)
    JA_PUNCT_RE,
    JapaneseTokenizer,
    edit_distance,
    error_rate,
    normalize_text,
)
from eval.biased_wer import BiasedWer, biased_wer  # noqa: E402
from eval.biased_wer import aggregate as aggregate_biased  # noqa: E402


def load_bias_terms(spec: Any) -> list[str]:
    """metrics.bias_terms を用語リストにする。

    受け付ける形:
      - 用語を直接並べたリスト
      - ファイルパス。1行1用語のテキスト、または1列目を用語とみなすTSV/CSV。
        '#' か '＃' で始まる行は無視する。
    """
    if not spec:
        return []
    if isinstance(spec, (list, tuple)):
        return [str(t).strip() for t in spec if str(t).strip()]

    path = Path(str(spec))
    if not path.exists():
        raise SystemExit(f"metrics.bias_terms のファイルがありません: {path}")

    rows = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith(("#", "＃"))
    ]
    if not rows:
        raise SystemExit(f"{path} から用語を読めませんでした")

    def split(line: str) -> list[str]:
        return [c.strip() for c in (line.split("\t") if "\t" in line else line.split(","))]

    # ヘッダに term / word 列があればその列を使う。無ければ1列目。
    # id 列が先頭にある表を素通しすると "T001" を用語として学習側に渡してしまう。
    header = [c.lower() for c in split(rows[0])]
    col = next((i for i, c in enumerate(header) if c in {"term", "word"}), None)
    if col is None:
        col, body = 0, rows
    else:
        body = rows[1:]

    terms = [cells[col] for line in body if (cells := split(line)) and len(cells) > col and cells[col]]
    if not terms:
        raise SystemExit(f"{path} から用語を読めませんでした")
    return terms

__all__ = [
    "JA_PUNCT_RE",
    "JapaneseTokenizer",
    "edit_distance",
    "error_rate",
    "normalize_text",
    "edit_operation_counts",
    "filter_supported_kwargs",
    "ngram_repetition_stats",
    "parse_temperature",
    "should_skip_no_speech_segment",
]


def filter_supported_kwargs(
    func: Callable[..., Any], kwargs: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """古い faster-whisper が未対応の keyword を signature に基づき除外する。"""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        # C拡張やdecoratorでsignatureが取れない場合は呼出し側でTypeErrorを明示化する。
        return dict(kwargs), []
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return dict(kwargs), []
    supported = {
        name
        for name, param in signature.parameters.items()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    unsupported = sorted(set(kwargs) - supported)
    return {key: value for key, value in kwargs.items() if key in supported}, unsupported


def parse_temperature(value: Any) -> float | list[float]:
    """YAML の scalar / sequence を faster-whisper の temperature 形式へ変換する。"""
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("temperature sequence must not be empty")
        return [float(item) for item in value]
    return float(value)


def optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def optional_positive_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive or null")
    return parsed


def should_skip_no_speech_segment(
    no_speech_prob: float,
    avg_logprob: float,
    *,
    no_speech_threshold: float | None,
    log_prob_threshold: float | None,
) -> bool:
    """faster-whisper と同じ複合条件で無音らしい低信頼 segment を除外する。"""
    if no_speech_threshold is None or no_speech_prob <= no_speech_threshold:
        return False
    return log_prob_threshold is None or avg_logprob < log_prob_threshold


def edit_operation_counts(ref: list[str], hyp: list[str]) -> dict[str, int]:
    """Levenshtein の置換・削除・挿入数を決定的な alignment で返す。"""
    # 各セルは (距離, 置換, 削除, 挿入)。2行だけ保持し、長い湧き出しでも
    # O(len(hyp)) メモリに抑える。
    previous = [(j, 0, 0, j) for j in range(len(hyp) + 1)]
    for i in range(1, len(ref) + 1):
        current = [(i, 0, i, 0)]
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                current.append(previous[j - 1])
                continue
            sub_cell = previous[j - 1]
            del_cell = previous[j]
            ins_cell = current[j - 1]
            # 同距離なら substitution -> deletion -> insertion の順で選び、結果を安定化する。
            candidates = [
                (sub_cell[0] + 1, sub_cell[1] + 1, sub_cell[2], sub_cell[3]),
                (del_cell[0] + 1, del_cell[1], del_cell[2] + 1, del_cell[3]),
                (ins_cell[0] + 1, ins_cell[1], ins_cell[2], ins_cell[3] + 1),
            ]
            current.append(min(candidates, key=lambda item: item[0]))
        previous = current

    _distance, substitutions, deletions, insertions = previous[-1]
    return {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
    }


def ngram_repetition_stats(units: list[str], ngram_size: int = 3) -> dict[str, int | float]:
    """仮説内で2回目以降に現れた n-gram の数と比率を返す。"""
    if ngram_size <= 0:
        raise ValueError("ngram_size must be positive")
    total = max(0, len(units) - ngram_size + 1)
    if not total:
        return {"total_ngrams": 0, "repeated_ngrams": 0, "repetition_ratio": 0.0}
    counts = Counter(tuple(units[index : index + ngram_size]) for index in range(total))
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return {
        "total_ngrams": total,
        "repeated_ngrams": repeated,
        "repetition_ratio": repeated / total,
    }


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
    condition_on_previous_text: bool = False
    use_initial_prompt: bool = False
    temperature: float | list[float] = 0.0
    no_speech_threshold: float | None = 0.6
    log_prob_threshold: float | None = -1.0
    compression_ratio_threshold: float | None = 2.4
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 3
    hallucination_silence_threshold: float | None = 2.0
    max_new_tokens: int | None = 192

    sep: str = ""
    _reported_unsupported: set[str] = field(default_factory=set, init=False, repr=False)

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
        options: dict[str, Any] = {
            "language": self.original_language,
            "task": self.task,
            "beam_size": self.beam_size,
            "word_timestamps": True,
            "condition_on_previous_text": self.condition_on_previous_text,
            "vad_filter": self.vad_filter,
            "temperature": self.temperature,
            "no_speech_threshold": self.no_speech_threshold,
            "log_prob_threshold": self.log_prob_threshold,
            "compression_ratio_threshold": self.compression_ratio_threshold,
            "repetition_penalty": self.repetition_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "hallucination_silence_threshold": self.hallucination_silence_threshold,
            "max_new_tokens": self.max_new_tokens,
        }
        if self.use_initial_prompt and init_prompt:
            options["initial_prompt"] = init_prompt
        options = {key: value for key, value in options.items() if value is not None}
        compatible_options, unsupported = filter_supported_kwargs(self.model.transcribe, options)
        new_unsupported = set(unsupported) - self._reported_unsupported
        if new_unsupported:
            print(
                "[WARN] installed faster-whisper does not support these decoding options; "
                f"ignoring them: {', '.join(sorted(new_unsupported))}",
                file=sys.stderr,
            )
            self._reported_unsupported.update(new_unsupported)
        try:
            segments, _info = self.model.transcribe(audio, **compatible_options)
        except TypeError as exc:
            raise RuntimeError(
                "faster-whisper transcribe API is incompatible with the configured options. "
                "Upgrade faster-whisper or remove unsupported decoding options from the config."
            ) from exc
        return list(segments)

    def ts_words(self, segments: list[Any]) -> list[tuple[float, float, str]]:
        out: list[tuple[float, float, str]] = []
        for segment in segments:
            if should_skip_no_speech_segment(
                float(getattr(segment, "no_speech_prob", 0.0)),
                float(getattr(segment, "avg_logprob", float("-inf"))),
                no_speech_threshold=self.no_speech_threshold,
                log_prob_threshold=self.log_prob_threshold,
            ):
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
        beam_size=int(model_cfg.get("beam_size", 3)),
        vad_filter=bool(ws_cfg.get("vad", True)),
        condition_on_previous_text=bool(ws_cfg.get("condition_on_previous_text", False)),
        use_initial_prompt=bool(ws_cfg.get("use_initial_prompt", False)),
        temperature=parse_temperature(ws_cfg.get("temperature", 0.0)),
        no_speech_threshold=optional_float(ws_cfg.get("no_speech_threshold", 0.6)),
        log_prob_threshold=optional_float(ws_cfg.get("log_prob_threshold", -1.0)),
        compression_ratio_threshold=optional_float(
            ws_cfg.get("compression_ratio_threshold", 2.4)
        ),
        repetition_penalty=float(ws_cfg.get("repetition_penalty", 1.1)),
        no_repeat_ngram_size=int(ws_cfg.get("no_repeat_ngram_size", 3)),
        hallucination_silence_threshold=optional_float(
            ws_cfg.get("hallucination_silence_threshold", 2.0)
        ),
        max_new_tokens=optional_positive_int(
            ws_cfg.get("max_new_tokens", 192), name="whisper_streaming.max_new_tokens"
        ),
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
    # metrics.bias_terms を指定すると B-WER / U-WER も出す。未指定なら従来通り。
    # 全体WERは専門用語1語を外しても僅かしか動かないので、用語だけを分母にした
    # B-WER を並べないとドメイン適応の効果が読めない。
    bias_terms = load_bias_terms(metrics_cfg.get("bias_terms"))
    repetition_ngram_size = int(metrics_cfg.get("repetition_ngram_size", 3))
    if repetition_ngram_size <= 0:
        raise SystemExit("metrics.repetition_ngram_size must be positive")
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
            edits = edit_operation_counts(ref_chars, hyp_chars)
            repetition = ngram_repetition_stats(hyp_chars, repetition_ngram_size)
            empty_ref_false_positive = not ref_chars and bool(hyp_chars)
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
                **(
                    biased_wer(
                        normalize_text(ref), normalize_text(hyp), bias_terms, tokenizer.words
                    ).as_dict()
                    if bias_terms
                    else {}
                ),
                "reference_chars": len(ref_chars),
                "prediction_chars": len(hyp_chars),
                "char_substitutions": edits["substitutions"],
                "char_deletions": edits["deletions"],
                "char_insertions": edits["insertions"],
                "empty_reference_false_positive": empty_ref_false_positive,
                "empty_reference_false_positive_chars": len(hyp_chars)
                if empty_ref_false_positive
                else 0,
                "repetition_ngram_size": repetition_ngram_size,
                "repeated_ngrams": repetition["repeated_ngrams"],
                "total_ngrams": repetition["total_ngrams"],
                "repetition_ratio": repetition["repetition_ratio"],
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
    total_reference_chars = sum(int(r["reference_chars"]) for r in results)
    total_prediction_chars = sum(int(r["prediction_chars"]) for r in results)
    total_insertions = sum(int(r["char_insertions"]) for r in results)
    total_deletions = sum(int(r["char_deletions"]) for r in results)
    total_substitutions = sum(int(r["char_substitutions"]) for r in results)
    empty_reference_count = sum(1 for r in results if int(r["reference_chars"]) == 0)
    empty_ref_fp_count = sum(1 for r in results if r["empty_reference_false_positive"])
    empty_ref_fp_chars = sum(int(r["empty_reference_false_positive_chars"]) for r in results)
    total_ngrams = sum(int(r["total_ngrams"]) for r in results)
    repeated_ngrams = sum(int(r["repeated_ngrams"]) for r in results)
    repetition_utterances = sum(1 for r in results if int(r["repeated_ngrams"]) > 0)
    summary = {
        "n": n,
        "manifest": str(manifest),
        "model_dir": model_dir,
        "tokenizer": tokenizer.mode,
        "cer": sum(r["cer"] for r in results) / n,
        "wer": sum(r["wer"] for r in results) / n,
        # 発話ごとの率を平均せず、誤り数と語数を合計してから割る。
        # 1語しかない短い発話が率を跳ねさせるのを混ぜないため。
        **(
            aggregate_biased(
                BiasedWer(
                    b_wer=None,
                    u_wer=None,
                    wer=None,
                    b_errors=int(r["b_errors"]),
                    b_total=int(r["b_total"]),
                    u_errors=int(r["u_errors"]),
                    u_total=int(r["u_total"]),
                )
                for r in results
            ).as_dict()
            if bias_terms
            else {}
        ),
        "reference_chars": total_reference_chars,
        "prediction_chars": total_prediction_chars,
        "char_substitutions": total_substitutions,
        "char_deletions": total_deletions,
        "char_insertions": total_insertions,
        "insertions": total_insertions,
        "insertion_rate": total_insertions / total_reference_chars
        if total_reference_chars
        else None,
        "empty_reference_count": empty_reference_count,
        "empty_reference_false_positive_count": empty_ref_fp_count,
        "empty_reference_false_positive_chars": empty_ref_fp_chars,
        # 短い別名も残し、後処理側で扱いやすくする。
        "empty_ref_false_positive_count": empty_ref_fp_count,
        "empty_ref_false_positive_chars": empty_ref_fp_chars,
        "repetition_ngram_size": repetition_ngram_size,
        "repeated_ngrams": repeated_ngrams,
        "repetition_count": repeated_ngrams,
        "total_ngrams": total_ngrams,
        "repetition_ratio": repeated_ngrams / total_ngrams if total_ngrams else 0.0,
        "repetition_utterance_count": repetition_utterances,
        "total_audio_sec": total_audio,
        "total_wall_sec": total_wall,
        "rtf": total_wall / total_audio if total_audio else None,
        "speed_x": total_audio / total_wall if total_wall else None,
        "min_chunk_size": min_chunk_size,
        "decoding": {
            "beam_size": asr.beam_size,
            "vad": asr.vad_filter,
            "condition_on_previous_text": asr.condition_on_previous_text,
            "use_initial_prompt": asr.use_initial_prompt,
            "temperature": asr.temperature,
            "no_speech_threshold": asr.no_speech_threshold,
            "log_prob_threshold": asr.log_prob_threshold,
            "compression_ratio_threshold": asr.compression_ratio_threshold,
            "repetition_penalty": asr.repetition_penalty,
            "no_repeat_ngram_size": asr.no_repeat_ngram_size,
            "hallucination_silence_threshold": asr.hallucination_silence_threshold,
            "max_new_tokens": asr.max_new_tokens,
        },
        "unsupported_faster_whisper_options": sorted(asr._reported_unsupported),
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
                f"- insertions: {summary['char_insertions']} / rate "
                + (
                    f"{summary['insertion_rate']:.4f}"
                    if summary["insertion_rate"] is not None
                    else "n/a"
                ),
                f"- empty-reference false positives: "
                f"{summary['empty_reference_false_positive_count']} utterances / "
                f"{summary['empty_reference_false_positive_chars']} chars",
                f"- repeated {summary['repetition_ngram_size']}-grams: "
                f"{summary['repeated_ngrams']} / {summary['total_ngrams']} "
                f"({summary['repetition_ratio']:.4f}); "
                f"utterances={summary['repetition_utterance_count']}",
                f"- unsupported faster-whisper options: "
                f"{summary['unsupported_faster_whisper_options'] or 'none'}",
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
