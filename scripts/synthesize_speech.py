#!/usr/bin/env python3
"""Synthesize a JSONL corpus with Qwen3-TTS or a vLLM-Omni server.

The input keeps two deliberately different strings:

``sentence``
    Original, normally kanji-containing, ASR training target.
``tts_text``
    Pronunciation-adjusted fallback text derived from the authoritative reading.

When ``--pronunciation-check-model`` is set, the first TTS attempt uses the
kanji-containing ``sentence`` for natural prosody.  Its audio is transcribed
without an initial prompt; only a failed/empty ASR check is synthesized again
with ``tts_text``.  A check failure therefore changes one item's TTS input but
does not stop the shard.  The output manifest preserves both strings and adds
``synthesis_text`` (the input that produced the adopted WAV).

Shards are selected deterministically with ``input_index % num_shards``.  Run
each shard with a separate ``--out-dir``.  A resumed run validates existing WAV
files and reconstructs the manifest in input order, so a truncated/stale
manifest never forces already completed audio to be synthesized again.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import io
import json
import logging
import os
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("synthesize_speech")

CONTROL_KINDS = frozenset({"silence", "noise"})
DEFAULT_CONTROL_SAMPLE_RATE = 24_000
PRONUNCIATION_MANIFEST_KEYS = (
    "pronunciation_check",
    "pronunciation_asr_text",
    "pronunciation_match",
    "pronunciation_check_reason",
    "pronunciation_expected_reading",
    "pronunciation_observed_reading",
    "pronunciation_mora_distance",
    "pronunciation_content_edits",
    "reading_fallback_used",
)


class AudioQualityError(ValueError):
    """Raised when generated audio is unsafe or unsuitable for training."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="例文 JSONL を Qwen3-TTS で高速バッチ音声合成する")
    parser.add_argument("--sentences", type=Path, required=True, help="例文 JSONL")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    # プリセット話者: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden,
    # Ono_Anna, Sohee。日本語ネイティブは Ono_Anna。
    parser.add_argument("--speaker", default="Ono_Anna", help="Qwen3-TTS プリセット話者名")
    parser.add_argument("--language", default="Japanese")
    parser.add_argument("--instruct", default=None, help="話し方のスタイル指示 (省略可)")
    parser.add_argument("--batch-size", type=int, default=8, help="Qwen/vLLM のバッチサイズ")
    parser.add_argument("--num-shards", type=int, default=1, help="入力を分割するシャード数")
    parser.add_argument("--shard-index", type=int, default=0, help="0 始まりの担当シャード")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="QC済みの既存WAVを再利用する (既定: true; --no-resume で無効)",
    )
    parser.add_argument(
        "--individual-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="バッチ失敗時に1件ずつ再試行する (既定: true)",
    )
    parser.add_argument("--lead-silence-ms", type=int, default=0, help="WAV先頭に付与する無音 (ミリ秒)")
    parser.add_argument("--trail-silence-ms", type=int, default=0, help="WAV末尾に付与する無音 (ミリ秒)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="auto: V100/Turing はfloat16、Ampere以降はbfloat16、CPUはfloat32",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        default="auto",
        help="auto: Ampere以降+flash_attnならFA2、それ以外(V100含む)はsdpa",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="vLLM-Omni URL。指定時は /v1/audio/speech/batch を使いローカルモデルを読まない",
    )
    parser.add_argument("--api-key", default=None, help="vLLM-Omni Bearer token (通常は不要)")
    parser.add_argument("--api-timeout", type=float, default=600.0, help="vLLM-Omni request timeout (秒)")
    parser.add_argument(
        "--pronunciation-check-model",
        default=None,
        help=(
            "faster-whisper model/path。指定時は漢字文を先に合成し、専門語を認識できない"
            "音声だけ読み仮名で再合成する"
        ),
    )
    parser.add_argument("--pronunciation-check-device", default="cuda")
    parser.add_argument("--pronunciation-check-device-index", type=int, default=0)
    parser.add_argument("--pronunciation-check-compute-type", default="float16")
    parser.add_argument("--pronunciation-check-beam-size", type=int, default=5)
    parser.add_argument(
        "--pronunciation-pass-threshold",
        type=float,
        default=0.15,
        help="許容変動を含む正規化mora距離の採用上限",
    )
    parser.add_argument(
        "--pronunciation-uncertain-threshold",
        type=float,
        default=0.35,
        help="pass上限を超えた候補をuncertainと記録する上限 (いずれもreading fallback)",
    )
    parser.add_argument("--control-sample-rate", type=int, default=DEFAULT_CONTROL_SAMPLE_RATE)
    parser.add_argument("--control-noise-rms", type=float, default=0.002)
    parser.add_argument("--min-duration-sec", type=float, default=0.10)
    parser.add_argument(
        "--max-duration-sec",
        type=float,
        default=30.0,
        help="過長な湧き出し音声を拒否する上限 (秒)",
    )
    parser.add_argument(
        "--duration-slack-sec",
        type=float,
        default=2.0,
        help="通常発話の文字数連動QCに足す固定余裕 (秒)",
    )
    parser.add_argument(
        "--max-seconds-per-char",
        type=float,
        default=0.45,
        help="通常発話の文1文字あたり最大秒数",
    )
    parser.add_argument("--min-rms", type=float, default=1e-5, help="通常発話の最小RMS")
    parser.add_argument(
        "--max-clip-fraction",
        type=float,
        default=0.01,
        help="abs(sample)>=0.999 の許容比率",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1:
        raise SystemExit("--batch-size は1以上にしてください")
    if args.num_shards < 1:
        raise SystemExit("--num-shards は1以上にしてください")
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index は 0 <= index < num-shards にしてください")
    if args.lead_silence_ms < 0 or args.trail_silence_ms < 0:
        raise SystemExit("無音長は0以上にしてください")
    if args.control_sample_rate < 1 or args.control_noise_rms < 0:
        raise SystemExit("control sample rate/RMS が不正です")
    if not 0 <= args.min_duration_sec < args.max_duration_sec:
        raise SystemExit("duration QC の範囲が不正です")
    if args.duration_slack_sec < 0 or args.max_seconds_per_char <= 0:
        raise SystemExit("文字数連動 duration QC の設定が不正です")
    if args.min_rms < 0 or not 0 <= args.max_clip_fraction <= 1:
        raise SystemExit("RMS/clip QC の範囲が不正です")
    if args.api_timeout <= 0:
        raise SystemExit("--api-timeout は0より大きくしてください")
    if args.pronunciation_check_device_index < 0:
        raise SystemExit("--pronunciation-check-device-index は0以上にしてください")
    if args.pronunciation_check_beam_size < 1:
        raise SystemExit("--pronunciation-check-beam-size は1以上にしてください")
    if not (
        0
        <= args.pronunciation_pass_threshold
        <= args.pronunciation_uncertain_threshold
        <= 1
    ):
        raise SystemExit(
            "発音確認thresholdは 0 <= pass <= uncertain <= 1 にしてください"
        )
    return args


def load_sentences(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_number}: JSONを解釈できません: {exc}") from exc
        if not isinstance(rec, dict):
            raise SystemExit(f"{path}:{line_number}: JSON object ではありません")
        if "id" not in rec:
            raise SystemExit(f"{path}:{line_number}: id がありません")
        term = str(rec.get("term", "") or "").strip()
        reading = str(rec.get("reading", "") or "").strip()
        sentence = sentence_text_for(rec)
        explicit = str(rec.get("tts_text", "") or "")
        if term and reading and term in sentence:
            derived = sentence.replace(term, reading)
            if explicit.strip() and explicit != derived:
                logger.warning(
                    "%s:%d id=%s: tts_text不一致。停止せず権威readingからfallbackを再計算します",
                    path,
                    line_number,
                    rec["id"],
                )
        records.append(rec)
    if not records:
        raise SystemExit(f"例文が見つかりません: {path}")
    return records


def select_shard(
    records: Sequence[dict[str, Any]], num_shards: int, shard_index: int
) -> list[dict[str, Any]]:
    """Return a stable, disjoint modulo shard without modifying records."""
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid shard configuration")
    return [rec for index, rec in enumerate(records) if index % num_shards == shard_index]


def _cuda_capability(torch: Any, device: str) -> tuple[int, int] | None:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return None
    try:
        return tuple(torch.cuda.get_device_capability(device))  # type: ignore[return-value]
    except (TypeError, RuntimeError, ValueError):
        # Some torch releases/fakes only accept an integer/current device.
        try:
            index = int(device.split(":", 1)[1]) if ":" in device else torch.cuda.current_device()
            return tuple(torch.cuda.get_device_capability(index))  # type: ignore[return-value]
        except (TypeError, RuntimeError, ValueError):
            return tuple(torch.cuda.get_device_capability())  # type: ignore[return-value]


def _flash_attn_available() -> bool:
    try:
        return importlib.util.find_spec("flash_attn") is not None
    except (ImportError, ValueError):
        return False


def resolve_model_runtime(args: argparse.Namespace, torch: Any) -> tuple[Any, str]:
    """Resolve a hardware-safe dtype and attention implementation."""
    capability = _cuda_capability(torch, str(args.device))
    requested_dtype = getattr(args, "dtype", "auto")
    if requested_dtype == "auto":
        if capability is None:
            dtype = torch.float32
        elif capability[0] >= 8:
            supports_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: True)()
            dtype = torch.bfloat16 if supports_bf16 else torch.float16
        else:
            dtype = torch.float16
    else:
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[requested_dtype]
        if capability is not None and capability[0] < 8 and requested_dtype == "bfloat16":
            logger.warning("compute capability %s はbfloat16非対応のためfloat16を使用", capability)
            dtype = torch.float16

    requested_attn = getattr(args, "attn_implementation", "auto")
    if requested_attn == "auto":
        attn = "flash_attention_2" if capability and capability[0] >= 8 and _flash_attn_available() else "sdpa"
    else:
        attn = requested_attn
    return dtype, attn


def load_model(args: argparse.Namespace) -> Any:
    import torch
    from qwen_tts import Qwen3TTSModel

    capability = _cuda_capability(torch, str(args.device))
    torch.backends.cuda.matmul.allow_tf32 = bool(capability and capability[0] >= 8)
    torch.backends.cudnn.allow_tf32 = bool(capability and capability[0] >= 8)
    dtype, attn = resolve_model_runtime(args, torch)
    logger.info(
        "Qwen3-TTS をロード中: %s (device=%s, dtype=%s, attention=%s)",
        args.model,
        args.device,
        dtype,
        attn,
    )
    kwargs: dict[str, Any] = {
        "device_map": args.device,
        "dtype": dtype,
        "attn_implementation": attn,
    }
    try:
        model = Qwen3TTSModel.from_pretrained(args.model, **kwargs)
    except TypeError as exc:
        # Compatibility with early qwen-tts packages that did not forward the
        # Transformers attention keyword.  Current releases take the first path.
        if "attn_implementation" not in str(exc):
            raise
        logger.warning("qwen-ttsがattn_implementation未対応のため旧APIで再試行")
        kwargs.pop("attn_implementation")
        model = Qwen3TTSModel.from_pretrained(args.model, **kwargs)
    logger.info("ロード完了")
    return model


def _as_audio_array(audio: Any) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach()
    if hasattr(audio, "cpu"):
        audio = audio.cpu()
    if hasattr(audio, "numpy"):
        audio = audio.numpy()
    result = np.asarray(audio, dtype=np.float32).squeeze()
    if result.ndim != 1:
        raise AudioQualityError(f"モノラルではない音声 shape={result.shape}")
    return result


def _split_wavs(wavs: Any, expected: int) -> list[Any]:
    if isinstance(wavs, (list, tuple)):
        result = list(wavs)
    elif expected == 1:
        result = [wavs]
    elif hasattr(wavs, "shape") and len(wavs) == expected:
        result = [wavs[index] for index in range(expected)]
    else:
        raise RuntimeError(f"Qwen3-TTS returned an unrecognized batch (expected {expected})")
    if len(result) != expected:
        raise RuntimeError(f"Qwen3-TTS returned {len(result)} WAVs for {expected} texts")
    return result


def _split_sample_rates(sample_rates: Any, expected: int) -> list[int]:
    if isinstance(sample_rates, (list, tuple, np.ndarray)) and not np.isscalar(sample_rates):
        rates = [int(value) for value in sample_rates]
        if len(rates) != expected:
            raise RuntimeError(f"TTS returned {len(rates)} sample rates for {expected} texts")
        return rates
    return [int(sample_rates)] * expected


def synthesize_batch(
    model: Any, args: argparse.Namespace, texts: Sequence[str]
) -> list[tuple[np.ndarray, int]]:
    """Use Qwen3-TTS's official list-valued custom-voice batch API."""
    if not texts:
        return []
    import torch

    count = len(texts)
    kwargs: dict[str, Any] = {
        "text": list(texts),
        "language": [args.language] * count,
        "speaker": [args.speaker] * count,
        # Keep this list-valued even when no style was requested.  This is the
        # public batch API shape and avoids a scalar being broadcast differently
        # by qwen-tts versions.
        "instruct": [args.instruct or ""] * count,
    }
    with torch.inference_mode():
        wavs, sample_rates = model.generate_custom_voice(**kwargs)
    split_wavs = _split_wavs(wavs, count)
    split_rates = _split_sample_rates(sample_rates, count)
    return [(_as_audio_array(audio), rate) for audio, rate in zip(split_wavs, split_rates)]


def synthesize(model: Any, args: argparse.Namespace, text: str) -> tuple[np.ndarray, int]:
    """Backward-compatible single-item wrapper (still calls the list API)."""
    return synthesize_batch(model, args, [text])[0]


def _batch_api_url(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/v1/audio/speech/batch"):
        return base
    if base.endswith("/v1"):
        return base + "/audio/speech/batch"
    return base + "/v1/audio/speech/batch"


def _decode_wav_bytes(encoded: str) -> tuple[np.ndarray, int]:
    if encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[-1]
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise RuntimeError("vLLM-Omni returned invalid base64 audio_data") from exc
    import soundfile as sf

    try:
        audio, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError("vLLM-Omni audio_data is not a readable WAV") from exc
    return _as_audio_array(audio), int(sample_rate)


def synthesize_batch_api(args: argparse.Namespace, texts: Sequence[str]) -> list[tuple[np.ndarray, int]]:
    """Call vLLM-Omni's OpenAI-compatible batch speech endpoint."""
    if not texts:
        return []
    items = [
        {
            "input": text,
            "voice": args.speaker,
            "language": args.language,
            "instructions": args.instruct or "",
            "response_format": "wav",
        }
        for text in texts
    ]
    body = json.dumps({"model": args.model, "items": items}, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if getattr(args, "api_key", None):
        headers["Authorization"] = f"Bearer {args.api_key}"
    request = urllib.request.Request(_batch_api_url(args.api_base), data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=args.api_timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM-Omni HTTP {exc.code}: {detail[:1000]}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"vLLM-Omni request failed: {exc}") from exc

    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list) or len(results) != len(texts):
        count = len(results) if isinstance(results, list) else "missing"
        raise RuntimeError(f"vLLM-Omni returned {count} results for {len(texts)} inputs")
    has_index = [isinstance(result, dict) and "index" in result for result in results]
    if any(has_index) and not all(has_index):
        raise RuntimeError("vLLM-Omni results contain only a partial set of indexes")

    ordered: list[dict[str, Any] | None] = [None] * len(texts)
    for response_position, result in enumerate(results):
        if not isinstance(result, dict):
            raise RuntimeError(f"vLLM-Omni result[{response_position}] is not an object")
        result_index = result.get("index", response_position)
        if isinstance(result_index, bool) or not isinstance(result_index, int):
            raise RuntimeError(f"vLLM-Omni result index is not an integer: {result_index!r}")
        if not 0 <= result_index < len(texts):
            raise RuntimeError(f"vLLM-Omni result index is out of range: {result_index}")
        if ordered[result_index] is not None:
            raise RuntimeError(f"vLLM-Omni returned duplicate result index: {result_index}")
        ordered[result_index] = result
    if any(result is None for result in ordered):
        raise RuntimeError("vLLM-Omni response is missing one or more result indexes")

    decoded: list[tuple[np.ndarray, int]] = []
    for result_index, result in enumerate(ordered):
        assert result is not None  # established by the complete-index check above
        if not isinstance(result.get("audio_data"), str):
            raise RuntimeError(
                f"vLLM-Omni result[{result_index}] has no audio_data: {result.get('error')}"
            )
        decoded.append(_decode_wav_bytes(result["audio_data"]))
    return decoded


def add_lead_silence(audio: np.ndarray, sample_rate: int, milliseconds: int) -> np.ndarray:
    """Add leading silence so playback/streaming does not clip the first phoneme."""
    samples = round(sample_rate * milliseconds / 1000)
    if samples <= 0:
        return audio
    return np.concatenate((np.zeros(samples, dtype=np.float32), audio.astype(np.float32, copy=False)))


def add_trail_silence(audio: np.ndarray, sample_rate: int, milliseconds: int) -> np.ndarray:
    """Add trailing silence so the final phoneme is not truncated."""
    samples = round(sample_rate * milliseconds / 1000)
    if samples <= 0:
        return audio
    return np.concatenate((audio.astype(np.float32, copy=False), np.zeros(samples, dtype=np.float32)))


def validate_audio(
    audio: Any,
    sample_rate: int,
    *,
    min_duration_sec: float = 0.10,
    max_duration_sec: float = 30.0,
    min_rms: float = 1e-5,
    max_clip_fraction: float = 0.01,
    allow_silence: bool = False,
) -> dict[str, float]:
    """Validate finite mono audio and return stable QC measurements."""
    array = _as_audio_array(audio)
    if sample_rate <= 0:
        raise AudioQualityError(f"invalid sample rate: {sample_rate}")
    if array.size == 0:
        raise AudioQualityError("empty audio")
    if not np.isfinite(array).all():
        raise AudioQualityError("audio contains NaN or infinity")
    duration = array.size / sample_rate
    if duration < min_duration_sec:
        raise AudioQualityError(f"audio is too short: {duration:.3f}s < {min_duration_sec:.3f}s")
    if duration > max_duration_sec:
        raise AudioQualityError(f"audio is too long: {duration:.3f}s > {max_duration_sec:.3f}s")
    # float64 accumulation prevents overflow/underflow from distorting QC.
    rms = float(np.sqrt(np.mean(np.square(array, dtype=np.float64))))
    if not allow_silence and rms < min_rms:
        raise AudioQualityError(f"audio RMS is too low: {rms:.8f} < {min_rms:.8f}")
    clip_fraction = float(np.mean(np.abs(array.astype(np.float64)) >= 0.999))
    if clip_fraction > max_clip_fraction:
        raise AudioQualityError(
            f"audio is clipped: fraction={clip_fraction:.6f} > {max_clip_fraction:.6f}"
        )
    return {"duration_sec": duration, "rms": rms, "clip_fraction": clip_fraction}


def make_control_audio(rec: dict[str, Any], args: argparse.Namespace) -> tuple[np.ndarray, int]:
    """Create deterministic no-speech controls without invoking Qwen."""
    kind = str(rec.get("kind", "")).strip().lower()
    if kind not in CONTROL_KINDS:
        raise ValueError(f"not a control record: {kind!r}")
    duration = float(rec.get("control_duration_sec", 1.0))
    if not np.isfinite(duration) or duration <= 0:
        raise AudioQualityError(f"invalid control_duration_sec: {duration!r}")
    sample_rate = int(args.control_sample_rate)
    sample_count = max(1, round(duration * sample_rate))
    if kind == "silence":
        return np.zeros(sample_count, dtype=np.float32), sample_rate

    seed = int(rec.get("control_seed", 0))
    target_rms = float(getattr(args, "control_noise_rms", 0.002))
    rng = np.random.default_rng(seed)
    audio = rng.standard_normal(sample_count).astype(np.float32)
    measured_rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    if measured_rms:
        audio *= np.float32(target_rms / measured_rms)
    return audio, sample_rate


def synthesis_text_for(rec: dict[str, Any]) -> str:
    """Return the reading-substituted fallback text without rejecting a row.

    The corpus generator normally supplies an exact ``tts_text`` already.  If
    the authoritative term and reading are present, derive the fallback again
    so a stale/missing ``tts_text`` never prevents the raw-kanji first attempt
    or stops a long synthesis job.
    """
    kind = str(rec.get("kind", "")).strip().lower()
    if kind in CONTROL_KINDS:
        return ""
    sentence = sentence_text_for(rec)
    tts_text = rec.get("tts_text")
    explicit_tts_text = (
        str(tts_text) if tts_text is not None and str(tts_text).strip() else None
    )

    term = str(rec.get("term", "") or "").strip()
    reading = str(rec.get("reading", "") or "").strip()
    if term and reading and term in sentence:
        return sentence.replace(term, reading)

    return explicit_tts_text if explicit_tts_text is not None else sentence


def sentence_text_for(rec: dict[str, Any]) -> str:
    if str(rec.get("kind", "")).strip().lower() in CONTROL_KINDS:
        return ""
    value = rec.get("sentence", rec.get("text", ""))
    return str(value) if value is not None else ""


def pronunciation_checkable(rec: dict[str, Any]) -> bool:
    if str(rec.get("kind", "")).strip().lower() in CONTROL_KINDS:
        return False
    term = str(rec.get("term", "") or "").strip()
    reading = str(rec.get("reading", "") or "").strip()
    return bool(term and reading and term in sentence_text_for(rec))


def preferred_synthesis_text_for(rec: dict[str, Any], *, checker_available: bool) -> str:
    """Prefer kanji only when the resulting audio can be checked and retried."""
    if checker_available and pronunciation_checkable(rec):
        return sentence_text_for(rec)
    return synthesis_text_for(rec)


def _compact_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return "".join(char for char in normalized if char.isalnum() or char in {"ー", "々"})


def _kata_to_hira(value: str) -> str:
    result: list[str] = []
    for char in value:
        code = ord(char)
        result.append(chr(code - 0x60) if 0x30A1 <= code <= 0x30F6 else char)
    return "".join(result)


def _morphological_reading(text: str, tagger: Any | None) -> str:
    if tagger is None:
        return ""
    pieces: list[str] = []
    try:
        for token in tagger(text):
            feature = getattr(token, "feature", None)
            reading = (
                getattr(feature, "kana", None)
                or getattr(feature, "pron", None)
                or getattr(feature, "reading", None)
            )
            pieces.append(str(reading or getattr(token, "surface", "")))
    except Exception:
        logger.debug("日本語形態素読みへの変換に失敗", exc_info=True)
        return ""
    return _kata_to_hira(_compact_text("".join(pieces)))


_SMALL_KANA = frozenset("ゃゅょぁぃぅぇぉゎゕゖ")
_MORA_VOWELS = {
    "a": frozenset("あかがさざただなはばぱまゃらわゎ"),
    "i": frozenset("いきぎしじちぢにひびぴみりゐぃ"),
    "u": frozenset("うくぐすずつづぬふぶぷむゆゅるゔぅ"),
    "e": frozenset("えけげせぜてでねへべぺめれゑぇ"),
    "o": frozenset("おこごそぞとどのほぼぽもよょろをぉ"),
}


def kana_to_morae(value: Any) -> list[str]:
    """Tokenize kana and collapse common Japanese long-vowel spellings.

    The normalized long marker carries its vowel (``ーe``/``ーo``), making
    セイ and セー equivalent without treating arbitrary mora substitutions as
    acceptable.
    """
    text = _kata_to_hira(unicodedata.normalize("NFKC", str(value or "")))
    raw: list[str] = []
    for char in text:
        if char in _SMALL_KANA and raw and raw[-1] not in {"っ", "ん", "ー"}:
            raw[-1] += char
        elif "ぁ" <= char <= "ゖ" or char in {"ー", "っ", "ん"}:
            raw.append(char)

    normalized: list[str] = []
    previous_vowel: str | None = None
    for mora in raw:
        vowel = _mora_vowel(mora)
        is_written_long = (
            mora == "ー"
            or (mora == "い" and previous_vowel == "e")
            or (mora == "う" and previous_vowel == "o")
            or (mora in {"あ", "い", "う", "え", "お"} and vowel == previous_vowel)
        )
        if is_written_long and previous_vowel is not None:
            normalized.append(f"ー{previous_vowel}")
            continue
        normalized.append(mora)
        previous_vowel = vowel
    return normalized


def _mora_vowel(mora: str) -> str | None:
    if not mora or mora in {"ー", "っ", "ん"}:
        return None
    for char in reversed(mora):
        for vowel, characters in _MORA_VOWELS.items():
            if char in characters:
                return vowel
    return None


def _gap_cost(mora: str) -> tuple[float, int]:
    # Long-vowel duration can be ambiguous in synthetic speech/ASR. A missing
    # or extra lexical mora (including ッ and ン) remains a content error.
    return (0.2, 0) if mora.startswith("ー") else (1.0, 1)


def weighted_mora_distance(
    expected: Sequence[str], observed: Sequence[str]
) -> tuple[float, int]:
    """Return ``(cost, substantive_edit_count)`` using tolerant long vowels."""
    inf = (float("inf"), 10**9)
    table: list[list[tuple[float, int]]] = [
        [inf for _ in range(len(observed) + 1)] for _ in range(len(expected) + 1)
    ]
    table[0][0] = (0.0, 0)

    def add(state: tuple[float, int], delta: tuple[float, int]) -> tuple[float, int]:
        return state[0] + delta[0], state[1] + delta[1]

    def best(*states: tuple[float, int]) -> tuple[float, int]:
        # Prefer an alignment with fewer real mora errors before breaking ties
        # by its total cost.
        return min(states, key=lambda state: (state[1], state[0]))

    for i in range(1, len(expected) + 1):
        table[i][0] = add(table[i - 1][0], _gap_cost(expected[i - 1]))
    for j in range(1, len(observed) + 1):
        table[0][j] = add(table[0][j - 1], _gap_cost(observed[j - 1]))
    for i, expected_mora in enumerate(expected, 1):
        for j, observed_mora in enumerate(observed, 1):
            substitution = (
                (0.0, 0) if expected_mora == observed_mora else (1.0, 1)
            )
            table[i][j] = best(
                add(table[i - 1][j], _gap_cost(expected_mora)),
                add(table[i][j - 1], _gap_cost(observed_mora)),
                add(table[i - 1][j - 1], substitution),
            )
    return table[-1][-1]


def best_mora_window(
    expected: Sequence[str], observed: Sequence[str]
) -> tuple[list[str], float, int]:
    """Find the transcript span most compatible with the target term reading."""
    if not expected or not observed:
        return [], 1.0, max(1, len(expected))
    minimum = max(1, min(len(observed), len(expected) - 2))
    maximum = min(len(observed), len(expected) + 2)
    candidates: list[tuple[int, float, list[str]]] = []
    for length in range(minimum, maximum + 1):
        for start in range(0, len(observed) - length + 1):
            window = list(observed[start : start + length])
            cost, content_edits = weighted_mora_distance(expected, window)
            normalized = cost / max(1, len(expected), len(window))
            candidates.append((content_edits, normalized, window))
    content_edits, normalized, window = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    return window, normalized, content_edits


def evaluate_pronunciation_transcript(
    rec: dict[str, Any],
    transcript: str,
    *,
    tagger: Any | None = None,
    pass_threshold: float = 0.15,
    uncertain_threshold: float = 0.35,
) -> dict[str, Any]:
    """Judge a target reading by tolerant mora distance, never raw text equality."""
    expected_reading = _kata_to_hira(_compact_text(rec.get("reading", "")))
    compact_transcript = _compact_text(transcript)
    if not compact_transcript:
        return {
            "passed": False,
            "transcript": transcript,
            "match": "",
            "reason": "empty_asr",
            "expected_reading": expected_reading,
            "observed_reading": "",
            "mora_distance": 1.0,
            "content_edits": max(1, len(kana_to_morae(expected_reading))),
        }

    direct_reading = _kata_to_hira(compact_transcript)
    observed_reading = _morphological_reading(transcript, tagger) or direct_reading
    expected_morae = kana_to_morae(expected_reading)
    observed_morae = kana_to_morae(observed_reading)
    window, distance, content_edits = best_mora_window(expected_morae, observed_morae)
    passed = content_edits == 0 and distance <= pass_threshold
    if passed:
        reason = ""
        match = "mora_exact" if distance == 0 else "mora_tolerant"
    elif content_edits:
        reason = "content_mora_mismatch"
        match = ""
    elif distance <= uncertain_threshold:
        reason = "uncertain_mora_score"
        match = ""
    else:
        reason = "mora_distance_too_large"
        match = ""
    return {
        "passed": passed,
        "transcript": transcript,
        "match": match,
        "reason": reason,
        "expected_reading": expected_reading,
        "observed_reading": observed_reading,
        "observed_term_morae": window,
        "mora_distance": round(distance, 6),
        "content_edits": content_edits,
    }


class PronunciationChecker:
    """Small faster-whisper wrapper used after the kanji-first TTS attempt."""

    def __init__(self, args: argparse.Namespace):
        from faster_whisper import WhisperModel

        model_kwargs: dict[str, Any] = {
            "device": args.pronunciation_check_device,
            "compute_type": args.pronunciation_check_compute_type,
        }
        if args.pronunciation_check_device == "cuda":
            model_kwargs["device_index"] = args.pronunciation_check_device_index
        logger.info(
            "発音確認ASRをロード中: %s (device=%s, compute_type=%s)",
            args.pronunciation_check_model,
            args.pronunciation_check_device,
            args.pronunciation_check_compute_type,
        )
        self.model = WhisperModel(args.pronunciation_check_model, **model_kwargs)
        self.beam_size = int(args.pronunciation_check_beam_size)
        self.pass_threshold = float(args.pronunciation_pass_threshold)
        self.uncertain_threshold = float(args.pronunciation_uncertain_threshold)
        try:
            from fugashi import Tagger

            self.tagger: Any | None = Tagger()
        except Exception:
            self.tagger = None
            logger.warning("fugashi辞書をロードできないため、ASR表記とかなだけで発音確認します")

    def check(
        self, audio: np.ndarray, sample_rate: int, rec: dict[str, Any]
    ) -> dict[str, Any]:
        import soundfile as sf

        try:
            wav = io.BytesIO()
            sf.write(wav, audio, sample_rate, format="WAV", subtype="PCM_16")
            wav.seek(0)
            segments, _info = self.model.transcribe(
                wav,
                language="ja",
                task="transcribe",
                beam_size=self.beam_size,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            transcript = "".join(str(segment.text) for segment in segments).strip()
            return evaluate_pronunciation_transcript(
                rec,
                transcript,
                tagger=self.tagger,
                pass_threshold=self.pass_threshold,
                uncertain_threshold=self.uncertain_threshold,
            )
        except Exception as exc:
            # Verification failure is deliberately soft: the caller will use
            # the authoritative reading fallback for this one record.
            logger.warning("%s: 発音確認ASRに失敗、読み仮名へfallback: %s", rec.get("id"), exc)
            return {
                "passed": False,
                "transcript": "",
                "match": "",
                "reason": f"asr_error:{type(exc).__name__}",
            }


def _safe_wav_path(wav_dir: Path, rec_id: Any) -> Path:
    value = str(rec_id)
    if not value or Path(value).name != value or value in {".", ".."}:
        raise SystemExit(f"WAVファイル名に使えない id です: {value!r}")
    return wav_dir / f"{value}.wav"


def max_allowed_duration(rec: dict[str, Any], args: argparse.Namespace) -> float:
    """Return the global/control limit or a tighter speech-text ratio limit."""
    if str(rec.get("kind", "")).strip().lower() in CONTROL_KINDS:
        return float(args.max_duration_sec)
    # Either the kanji-first candidate or the reading fallback can be adopted.
    text_length = max(1, len(sentence_text_for(rec)), len(synthesis_text_for(rec)))
    text_limit = args.duration_slack_sec + args.max_seconds_per_char * text_length
    return float(min(args.max_duration_sec, text_limit))


def _quality_kwargs(args: argparse.Namespace, rec: dict[str, Any]) -> dict[str, float]:
    return {
        "min_duration_sec": args.min_duration_sec,
        "max_duration_sec": max_allowed_duration(rec, args),
        "min_rms": args.min_rms,
        "max_clip_fraction": args.max_clip_fraction,
    }


def _pad_and_validate(
    audio: Any, sample_rate: int, rec: dict[str, Any], args: argparse.Namespace
) -> tuple[np.ndarray, dict[str, float]]:
    array = _as_audio_array(audio)
    array = add_lead_silence(array, sample_rate, args.lead_silence_ms)
    array = add_trail_silence(array, sample_rate, args.trail_silence_ms)
    is_control = str(rec.get("kind", "")).strip().lower() in CONTROL_KINDS
    quality = validate_audio(array, sample_rate, allow_silence=is_control, **_quality_kwargs(args, rec))
    return array, quality


def _read_valid_wav(
    path: Path, rec: dict[str, Any], args: argparse.Namespace
) -> tuple[np.ndarray, int, dict[str, float]]:
    import soundfile as sf

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    is_control = str(rec.get("kind", "")).strip().lower() in CONTROL_KINDS
    quality = validate_audio(
        audio, int(sample_rate), allow_silence=is_control, **_quality_kwargs(args, rec)
    )
    return _as_audio_array(audio), int(sample_rate), quality


def _atomic_write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        sf.write(temporary, audio, sample_rate, format="WAV", subtype="PCM_16")
        # Windows rejects fsync on a read-only descriptor (Linux accepts it),
        # so use a read/write handle for portable durability.
        with temporary.open("r+b") as fh:
            os.fsync(fh.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_manifest_entry(
    rec: dict[str, Any],
    wav_path: Path,
    out_dir: Path,
    sample_rate: int,
    quality: dict[str, float],
    args: argparse.Namespace,
    *,
    synthesis_text: str | None = None,
    pronunciation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Preserve source metadata while keeping target and synthesis text apart."""
    entry = dict(rec)
    original_sentence = rec.get("sentence", rec.get("text", ""))
    original_sentence = "" if original_sentence is None else str(original_sentence)
    fallback_text = synthesis_text_for(rec)
    actual_synthesis_text = fallback_text if synthesis_text is None else synthesis_text
    entry.update(
        {
            "id": rec["id"],
            "kind": rec.get("kind", "speech"),
            "sentence": original_sentence,
            "tts_text": fallback_text,
            "synthesis_text": actual_synthesis_text,
            "wav": str(wav_path.relative_to(out_dir)),
            "duration_sec": round(quality["duration_sec"], 3),
            "sample_rate": int(sample_rate),
            "model": args.model,
            "speaker": args.speaker,
            "language": args.language,
            "instruct": args.instruct or "",
            "lead_silence_ms": int(args.lead_silence_ms),
            "trail_silence_ms": int(args.trail_silence_ms),
            "synthesis_fingerprint": synthesis_fingerprint(rec, args),
        }
    )
    if pronunciation:
        entry.update(pronunciation)
    return entry


def synthesis_fingerprint(rec: dict[str, Any], args: argparse.Namespace) -> str:
    """Fingerprint every setting that can change audio or its ASR label."""
    payload = {
        "version": 2,
        "kind": str(rec.get("kind", "speech")),
        "sentence": str(rec.get("sentence", rec.get("text", "")) or ""),
        "tts_text": str(rec.get("tts_text", "") or ""),
        "reading_fallback_text": synthesis_text_for(rec),
        "kanji_candidate_text": sentence_text_for(rec),
        "model": str(args.model),
        "speaker": str(args.speaker),
        "language": str(args.language),
        "instruct": str(args.instruct or ""),
        "backend": "vllm-omni" if args.api_base else "qwen-tts",
        "dtype": str(args.dtype),
        "attn_implementation": str(args.attn_implementation),
        "lead_silence_ms": int(args.lead_silence_ms),
        "trail_silence_ms": int(args.trail_silence_ms),
        "control_duration_sec": rec.get("control_duration_sec"),
        "control_seed": rec.get("control_seed"),
        "control_sample_rate": int(args.control_sample_rate),
        "control_noise_rms": float(args.control_noise_rms),
        "pronunciation_check_model": str(
            getattr(args, "pronunciation_check_model", None) or ""
        ),
        "pronunciation_check_device": str(
            getattr(args, "pronunciation_check_device", "cuda")
        ),
        "pronunciation_check_compute_type": str(
            getattr(args, "pronunciation_check_compute_type", "float16")
        ),
        "pronunciation_check_beam_size": int(
            getattr(args, "pronunciation_check_beam_size", 5)
        ),
        "pronunciation_pass_threshold": float(
            getattr(args, "pronunciation_pass_threshold", 0.15)
        ),
        "pronunciation_uncertain_threshold": float(
            getattr(args, "pronunciation_uncertain_threshold", 0.35)
        ),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_previous_manifest(path: Path) -> dict[str, dict[str, Any]]:
    """Load every intact row; tolerate a truncated tail from a legacy run."""
    entries: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return entries
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("旧manifestの壊れた行を無視: %s:%d (%s)", path, line_number, exc)
            continue
        if isinstance(entry, dict) and "id" in entry:
            entries[str(entry["id"])] = entry
    return entries


def provenance_matches(rec: dict[str, Any], old_entry: dict[str, Any] | None, args: argparse.Namespace) -> bool:
    """Only reuse WAVs whose exact source/config fingerprint is known."""
    fingerprint_matches = bool(
        old_entry
        and old_entry.get("synthesis_fingerprint")
        and old_entry.get("synthesis_fingerprint") == synthesis_fingerprint(rec, args)
    )
    if not fingerprint_matches:
        return False
    if getattr(args, "pronunciation_check_model", None) and pronunciation_checkable(rec):
        # A previous soft fallback caused by a missing ASR runtime should be
        # retried next time rather than becoming a permanent reading-only WAV.
        return old_entry.get("pronunciation_check") in {"passed", "fallback_reading"}
    return True


def write_manifest_atomic(
    manifest_path: Path,
    selected_records: Sequence[dict[str, Any]],
    entries: dict[str, dict[str, Any]],
) -> None:
    """Atomically rebuild the manifest in stable input order and fsync it."""
    temporary = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as fh:
            for rec in selected_records:
                entry = entries.get(str(rec["id"]))
                if entry is not None:
                    fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, manifest_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _call_backend(
    model: Any | None, args: argparse.Namespace, texts: Sequence[str]
) -> list[tuple[np.ndarray, int]]:
    if args.api_base:
        return synthesize_batch_api(args, texts)
    if model is None:
        raise RuntimeError("Qwen3-TTS model is not loaded")
    return synthesize_batch(model, args, texts)


def _iter_batches(records: Sequence[dict[str, Any]], size: int) -> Sequence[dict[str, Any]]:
    for start in range(0, len(records), size):
        yield records[start : start + size]


def run(args: argparse.Namespace) -> None:
    records = load_sentences(args.sentences)
    selected = select_shard(records, args.num_shards, args.shard_index)
    # vLLM-Omni's public batch schema accepts at most 32 items per request.
    # Keep the CLI value unrestricted for the official in-process Qwen backend.
    backend_batch_size = min(args.batch_size, 32) if args.api_base else args.batch_size
    if backend_batch_size != args.batch_size:
        logger.warning("vLLM-Omniの上限によりbatch-sizeを32に分割します")
    logger.info(
        "全%d件から shard %d/%d の%d件を合成します (batch=%d)",
        len(records),
        args.shard_index,
        args.num_shards,
        len(selected),
        backend_batch_size,
    )

    wav_dir = args.out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.jsonl"
    previous_entries = load_previous_manifest(manifest_path) if args.resume else {}
    entries: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for rec in selected:
        rec_id = str(rec["id"])
        if rec_id in seen_ids:
            raise SystemExit(f"担当シャード内で id が重複しています: {rec_id}")
        seen_ids.add(rec_id)
        wav_path = _safe_wav_path(wav_dir, rec_id)
        old_entry = previous_entries.get(rec_id)
        if args.resume and wav_path.is_file() and provenance_matches(rec, old_entry, args):
            try:
                _audio, sample_rate, quality = _read_valid_wav(wav_path, rec, args)
            except (AudioQualityError, OSError, RuntimeError, ValueError) as exc:
                logger.warning("既存WAVを再生成します: %s (%s)", wav_path, exc)
            else:
                old_pronunciation = {
                    key: old_entry[key]
                    for key in PRONUNCIATION_MANIFEST_KEYS
                    if key in old_entry
                }
                entries[rec_id] = build_manifest_entry(
                    rec,
                    wav_path,
                    args.out_dir,
                    sample_rate,
                    quality,
                    args,
                    synthesis_text=str(
                        old_entry.get("synthesis_text", synthesis_text_for(rec))
                    ),
                    pronunciation=old_pronunciation,
                )
                continue
        elif args.resume and wav_path.is_file():
            logger.info("既存WAVの生成条件が不明/不一致のため再生成: %s", wav_path)
        pending.append(rec)

    # Rebuild immediately.  This repairs truncated/old manifests and makes a
    # --no-resume run visibly start from zero completed records.
    write_manifest_atomic(manifest_path, selected, entries)
    resumed = len(entries)
    if resumed:
        logger.info("QC済み既存WAV %d件をスキップ", resumed)
    if not pending:
        logger.info("完了: %s", manifest_path)
        return

    needs_model = any(
        str(rec.get("kind", "")).strip().lower() not in CONTROL_KINDS for rec in pending
    )
    checker: PronunciationChecker | None = None
    checker_load_reason = ""
    checker_requested = bool(args.pronunciation_check_model) and any(
        pronunciation_checkable(rec) for rec in pending
    )
    if checker_requested:
        try:
            checker = PronunciationChecker(args)
        except Exception as exc:
            checker_load_reason = f"checker_load_error:{type(exc).__name__}"
            logger.warning(
                "発音確認ASRをロードできません。停止せず、このshardは権威readingで合成します: %s",
                exc,
            )

    model = None if args.api_base or not needs_model else load_model(args)
    failures: list[tuple[str, str]] = []
    start_time = time.monotonic()
    processed_new = 0
    pronunciation_counts = {"passed": 0, "fallback_reading": 0, "checker_unavailable": 0}

    def commit_one(
        rec: dict[str, Any],
        audio: np.ndarray,
        sample_rate: int,
        quality: dict[str, float],
        synthesis_text: str,
        pronunciation: dict[str, Any] | None = None,
    ) -> None:
        nonlocal processed_new
        rec_id = str(rec["id"])
        wav_path = _safe_wav_path(wav_dir, rec_id)
        _atomic_write_wav(wav_path, audio, sample_rate)
        entries[rec_id] = build_manifest_entry(
            rec,
            wav_path,
            args.out_dir,
            sample_rate,
            quality,
            args,
            synthesis_text=synthesis_text,
            pronunciation=pronunciation,
        )
        if pronunciation:
            status = str(pronunciation.get("pronunciation_check", ""))
            if status in pronunciation_counts:
                pronunciation_counts[status] += 1
        processed_new += 1
        write_manifest_atomic(manifest_path, selected, entries)
        elapsed = time.monotonic() - start_time
        done = resumed + processed_new
        remaining = len(selected) - done
        eta = elapsed / processed_new * remaining if processed_new else 0.0
        logger.info(
            "(%d/%d) %s [%.2fs] %s | 経過 %.0f秒 / 残り目安 %.0f秒",
            done,
            len(selected),
            wav_path.name,
            quality["duration_sec"],
            synthesis_text[:30],
            elapsed,
            eta,
        )

    def finish_one(
        rec: dict[str, Any],
        audio: Any,
        sample_rate: int,
        synthesis_text: str,
        pronunciation: dict[str, Any] | None = None,
    ) -> None:
        array, quality = _pad_and_validate(audio, sample_rate, rec, args)
        commit_one(
            rec,
            array,
            sample_rate,
            quality,
            synthesis_text,
            pronunciation,
        )

    def adopt_candidate(
        rec: dict[str, Any], audio: Any, sample_rate: int, candidate_text: str
    ) -> None:
        """Adopt a checked kanji candidate or transparently synthesize its fallback."""
        array, quality = _pad_and_validate(audio, sample_rate, rec, args)
        pronunciation: dict[str, Any] | None = None

        if checker is not None and pronunciation_checkable(rec):
            result = checker.check(array, sample_rate, rec)
            pronunciation = {
                "pronunciation_asr_text": result.get("transcript", ""),
                "pronunciation_match": result.get("match", ""),
                "pronunciation_check_reason": result.get("reason", ""),
                "pronunciation_expected_reading": result.get(
                    "expected_reading", rec.get("reading", "")
                ),
                "pronunciation_observed_reading": result.get("observed_reading", ""),
                "pronunciation_mora_distance": result.get("mora_distance"),
                "pronunciation_content_edits": result.get("content_edits"),
            }
            if result.get("passed"):
                pronunciation.update(
                    {"pronunciation_check": "passed", "reading_fallback_used": False}
                )
                logger.info(
                    "%s: 漢字TTSの読みを採用 "
                    "(ASR=%r, match=%s, mora_distance=%s)",
                    rec["id"],
                    result.get("transcript", ""),
                    result.get("match", ""),
                    result.get("mora_distance"),
                )
                commit_one(
                    rec,
                    array,
                    sample_rate,
                    quality,
                    candidate_text,
                    pronunciation,
                )
                return

            fallback_text = synthesis_text_for(rec)
            logger.warning(
                "%s: 漢字TTSの読みを確認できないため、この1件だけreadingへfallback "
                "(ASR=%r, reason=%s, mora_distance=%s, content_edits=%s)",
                rec["id"],
                result.get("transcript", ""),
                result.get("reason", ""),
                result.get("mora_distance"),
                result.get("content_edits"),
            )
            fallback_audio, fallback_rate = _call_backend(model, args, [fallback_text])[0]
            pronunciation.update(
                {
                    "pronunciation_check": "fallback_reading",
                    "reading_fallback_used": True,
                }
            )
            finish_one(
                rec,
                fallback_audio,
                fallback_rate,
                fallback_text,
                pronunciation,
            )
            return

        if checker_requested and pronunciation_checkable(rec):
            # Loading the verifier is also soft-fail. Since this candidate was
            # generated from the fallback text, it is safe to keep.
            pronunciation = {
                "pronunciation_check": "checker_unavailable",
                "pronunciation_asr_text": "",
                "pronunciation_match": "",
                "pronunciation_check_reason": checker_load_reason,
                "pronunciation_expected_reading": rec.get("reading", ""),
                "pronunciation_observed_reading": "",
                "pronunciation_mora_distance": None,
                "pronunciation_content_edits": None,
                "reading_fallback_used": True,
            }
        commit_one(
            rec,
            array,
            sample_rate,
            quality,
            candidate_text,
            pronunciation,
        )

    for batch in _iter_batches(pending, backend_batch_size):
        speech_records: list[dict[str, Any]] = []
        for rec in batch:
            kind = str(rec.get("kind", "")).strip().lower()
            if kind in CONTROL_KINDS:
                try:
                    audio, sample_rate = make_control_audio(rec, args)
                    finish_one(rec, audio, sample_rate, "")
                except Exception as exc:  # keep completed records resumable
                    logger.error("control %s の生成に失敗: %s", rec["id"], exc)
                    failures.append((str(rec["id"]), str(exc)))
            else:
                text = preferred_synthesis_text_for(
                    rec, checker_available=checker is not None
                )
                if not text.strip():
                    failures.append((str(rec["id"]), "synthesis text is empty"))
                    logger.error("%s: TTS入力が空です", rec["id"])
                else:
                    speech_records.append(rec)

        if not speech_records:
            continue
        texts = [
            preferred_synthesis_text_for(rec, checker_available=checker is not None)
            for rec in speech_records
        ]
        try:
            outputs = _call_backend(model, args, texts)
        except Exception as batch_exc:
            if not args.individual_fallback or len(speech_records) == 1:
                logger.error("バッチ合成失敗: %s", batch_exc)
                failures.extend((str(rec["id"]), str(batch_exc)) for rec in speech_records)
                continue
            logger.warning("バッチ合成失敗、1件ずつ再試行します: %s", batch_exc)
            outputs = []
            for rec, text in zip(speech_records, texts):
                try:
                    output = _call_backend(model, args, [text])[0]
                    adopt_candidate(rec, *output, text)
                except Exception as exc:
                    logger.error("%s の個別合成に失敗: %s", rec["id"], exc)
                    failures.append((str(rec["id"]), str(exc)))
            continue

        for rec, output, text in zip(speech_records, outputs, texts):
            try:
                adopt_candidate(rec, *output, text)
            except Exception as first_exc:
                # Batch decoding can occasionally produce one malformed/overlong
                # item.  Retry only that item before declaring it failed.
                if args.individual_fallback and len(speech_records) > 1:
                    try:
                        retry = _call_backend(model, args, [text])[0]
                        adopt_candidate(rec, *retry, text)
                        continue
                    except Exception as retry_exc:
                        message = f"batch output: {first_exc}; individual retry: {retry_exc}"
                else:
                    message = str(first_exc)
                logger.error("%s のWAV/QC処理に失敗: %s", rec["id"], message)
                failures.append((str(rec["id"]), message))

    if failures:
        preview = "; ".join(f"{rec_id}: {message}" for rec_id, message in failures[:5])
        raise SystemExit(f"音声合成に {len(failures)} 件失敗しました ({preview})")
    if checker_requested:
        logger.info(
            "発音確認 summary: kanji採用=%d reading fallback=%d checker利用不可=%d",
            pronunciation_counts["passed"],
            pronunciation_counts["fallback_reading"],
            pronunciation_counts["checker_unavailable"],
        )
    logger.info("完了: %s", manifest_path)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
