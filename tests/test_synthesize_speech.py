import base64
import io
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from scripts import synthesize_speech as tts


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_qwen_batch_uses_list_valued_official_api(monkeypatch):
    calls = []

    class InferenceMode:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return False

    fake_torch = SimpleNamespace(inference_mode=lambda: InferenceMode())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class Model:
        def generate_custom_voice(self, **kwargs):
            calls.append(kwargs)
            return [np.full(400, 0.1), np.full(600, 0.2)], 24_000

    args = SimpleNamespace(language="Japanese", speaker="Ono_Anna", instruct=None)
    outputs = tts.synthesize_batch(Model(), args, ["一つ", "二つ"])

    assert calls == [
        {
            "text": ["一つ", "二つ"],
            "language": ["Japanese", "Japanese"],
            "speaker": ["Ono_Anna", "Ono_Anna"],
            "instruct": ["", ""],
        }
    ]
    assert [audio.shape for audio, _sr in outputs] == [(400,), (600,)]
    assert [sr for _audio, sr in outputs] == [24_000, 24_000]


def test_runtime_auto_uses_fp16_sdpa_on_v100_and_fa2_on_ampere(monkeypatch):
    class FakeCuda:
        capability = (7, 0)

        @staticmethod
        def is_available():
            return True

        @classmethod
        def get_device_capability(cls, *_args):
            return cls.capability

        @staticmethod
        def is_bf16_supported():
            return True

    torch = SimpleNamespace(
        cuda=FakeCuda,
        float16="fp16",
        bfloat16="bf16",
        float32="fp32",
    )
    args = SimpleNamespace(device="cuda:0", dtype="auto", attn_implementation="auto")
    monkeypatch.setattr(tts, "_flash_attn_available", lambda: True)

    assert tts.resolve_model_runtime(args, torch) == ("fp16", "sdpa")
    FakeCuda.capability = (8, 0)
    assert tts.resolve_model_runtime(args, torch) == ("bf16", "flash_attention_2")


def test_vllm_batch_api_payload_and_wav_decode(monkeypatch):
    def encoded_wav(sample_count):
        buffer = io.BytesIO()
        sf.write(
            buffer,
            np.full(sample_count, 0.1, dtype=np.float32),
            24_000,
            format="WAV",
            subtype="PCM_16",
        )
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    encoded_first = encoded_wav(1_200)
    encoded_second = encoded_wav(2_400)
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            # vLLM-Omni may finish fan-out requests out of order.  The client
            # must restore input order using results[].index.
            return json.dumps(
                {
                    "results": [
                        {"index": 1, "audio_data": encoded_second},
                        {"index": 0, "audio_data": encoded_first},
                    ]
                }
            ).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(tts.urllib.request, "urlopen", fake_urlopen)
    args = SimpleNamespace(
        api_base="http://127.0.0.1:8000/v1",
        api_key=None,
        api_timeout=42,
        model="Qwen/Qwen3-TTS",
        speaker="Ono_Anna",
        language="Japanese",
        instruct="ゆっくり",
    )

    outputs = tts.synthesize_batch_api(args, ["試験文", "二つ目"])
    assert captured["url"] == "http://127.0.0.1:8000/v1/audio/speech/batch"
    assert captured["timeout"] == 42
    assert captured["body"] == {
        "model": "Qwen/Qwen3-TTS",
        "items": [
            {
                "input": "試験文",
                "voice": "Ono_Anna",
                "language": "Japanese",
                "instructions": "ゆっくり",
                "response_format": "wav",
            },
            {
                "input": "二つ目",
                "voice": "Ono_Anna",
                "language": "Japanese",
                "instructions": "ゆっくり",
                "response_format": "wav",
            },
        ],
    }
    assert [sample_rate for _audio, sample_rate in outputs] == [24_000, 24_000]
    assert [audio.shape for audio, _sample_rate in outputs] == [(1_200,), (2_400,)]


def test_vllm_batch_api_rejects_duplicate_indexes(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"results": [{"index": 0}, {"index": 0}]}).encode()

    monkeypatch.setattr(tts.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    args = SimpleNamespace(
        api_base="http://127.0.0.1:8000",
        api_key=None,
        api_timeout=42,
        model="Qwen/Qwen3-TTS",
        speaker="Ono_Anna",
        language="Japanese",
        instruct=None,
    )
    with pytest.raises(RuntimeError, match="duplicate result index"):
        tts.synthesize_batch_api(args, ["一つ", "二つ"])


def test_resume_rebuilds_stable_manifest_and_keeps_kanji_target(tmp_path, monkeypatch):
    source = tmp_path / "sentences.jsonl"
    output = tmp_path / "audio"
    records = [
        {
            "id": "a",
            "kind": "speech",
            "term": "専門用語",
            "source_term": "専門用語",
            "reading": "せんもんようご",
            "sentence": "専門用語を認識します。",
            "tts_text": "せんもんようごを認識します。",
        },
        {
            "id": "b",
            "kind": "speech",
            "term": "音声認識",
            "reading": "おんせいにんしき",
            "sentence": "音声認識を評価します。",
            "tts_text": "おんせいにんしきを評価します。",
        },
    ]
    _write_jsonl(source, records)
    args = tts.parse_args(
        [
            "--sentences",
            str(source),
            "--out-dir",
            str(output),
            "--batch-size",
            "2",
            "--min-duration-sec",
            "0.01",
        ]
    )
    batches = []
    monkeypatch.setattr(tts, "load_model", lambda _args: object())

    def fake_backend(_model, _args, texts):
        batches.append(list(texts))
        return [(np.full(4_800, 0.1, dtype=np.float32), 24_000) for _ in texts]

    monkeypatch.setattr(tts, "_call_backend", fake_backend)
    tts.run(args)
    assert batches == [[records[0]["tts_text"], records[1]["tts_text"]]]

    manifest_path = output / "manifest.jsonl"
    first_manifest = manifest_path.read_text(encoding="utf-8")
    manifest = [json.loads(line) for line in first_manifest.splitlines()]
    assert [row["id"] for row in manifest] == ["a", "b"]
    assert manifest[0]["sentence"] == records[0]["sentence"]
    assert manifest[0]["tts_text"] == records[0]["tts_text"]
    assert manifest[0]["synthesis_text"] == records[0]["tts_text"]
    assert manifest[0]["source_term"] == "専門用語"

    # Stable reconstruction restores input order without re-synthesizing when
    # both WAV QC and the source/config fingerprint match.
    manifest_path.write_text(
        "\n".join(reversed(first_manifest.splitlines())) + "\n", encoding="utf-8"
    )
    batches.clear()
    tts.run(args)
    assert not batches
    assert manifest_path.read_text(encoding="utf-8") == first_manifest

    # Reusing the same stable ID with different synthesis text must not attach
    # the old audio to a new label.
    changed = [dict(row) for row in records]
    changed[0]["sentence"] = "専門用語をもう一度認識します。"
    changed[0]["tts_text"] = "せんもんようごをもう一度認識します。"
    _write_jsonl(source, changed)
    batches.clear()
    tts.run(args)
    assert batches == [[changed[0]["tts_text"]]]
    changed_manifest = [
        json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    assert changed_manifest[0]["sentence"] == changed[0]["sentence"]
    assert changed_manifest[0]["synthesis_fingerprint"] != manifest[0]["synthesis_fingerprint"]


def test_control_audio_is_deterministic_and_does_not_load_qwen(tmp_path, monkeypatch):
    source = tmp_path / "controls.jsonl"
    output = tmp_path / "audio"
    records = [
        {
            "id": "silence",
            "kind": "silence",
            "term": "",
            "reading": "",
            "source_term": "",
            "sentence": "",
            "tts_text": "",
            "control_duration_sec": 0.2,
            "control_seed": 1,
        },
        {
            "id": "noise",
            "kind": "noise",
            "term": "",
            "reading": "",
            "source_term": "",
            "sentence": "",
            "tts_text": "",
            "control_duration_sec": 0.2,
            "control_seed": 123,
        },
    ]
    _write_jsonl(source, records)
    args = tts.parse_args(
        [
            "--sentences",
            str(source),
            "--out-dir",
            str(output),
            "--lead-silence-ms",
            "10",
            "--trail-silence-ms",
            "10",
            "--no-resume",
        ]
    )
    monkeypatch.setattr(tts, "load_model", lambda _args: pytest.fail("Qwen must not be loaded"))

    tts.run(args)
    first_noise = (output / "wav" / "noise.wav").read_bytes()
    tts.run(args)
    assert (output / "wav" / "noise.wav").read_bytes() == first_noise

    silence, sample_rate = sf.read(output / "wav" / "silence.wav", dtype="float32")
    noise, _ = sf.read(output / "wav" / "noise.wav", dtype="float32")
    assert sample_rate == 24_000
    assert silence.shape == noise.shape == (5_280,)
    assert np.all(silence == 0)
    assert 0.0015 < np.sqrt(np.mean(noise.astype(np.float64) ** 2)) < 0.0021
    manifest = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert manifest[0]["sentence"] == manifest[0]["tts_text"] == manifest[0]["synthesis_text"] == ""
    assert manifest[1]["control_seed"] == 123


def test_audio_qc_rejects_nonfinite_overlong_quiet_and_clipped_audio():
    kwargs = dict(min_duration_sec=0.01, max_duration_sec=1.0, min_rms=1e-4, max_clip_fraction=0.01)
    with pytest.raises(tts.AudioQualityError, match="NaN"):
        tts.validate_audio(np.array([np.nan] * 100), 1_000, **kwargs)
    with pytest.raises(tts.AudioQualityError, match="too long"):
        tts.validate_audio(np.full(1_100, 0.1), 1_000, **kwargs)
    with pytest.raises(tts.AudioQualityError, match="RMS"):
        tts.validate_audio(np.zeros(100), 1_000, **kwargs)
    with pytest.raises(tts.AudioQualityError, match="clipped"):
        tts.validate_audio(np.ones(100), 1_000, **kwargs)
    quality = tts.validate_audio(np.zeros(100), 1_000, allow_silence=True, **kwargs)
    assert quality["rms"] == 0


def test_speech_duration_qc_scales_with_text_but_controls_use_global_limit():
    args = SimpleNamespace(
        max_duration_sec=30.0,
        duration_slack_sec=2.0,
        max_seconds_per_char=0.45,
    )
    speech = {"kind": "speech", "sentence": "あ" * 10, "tts_text": "あ" * 10}
    control = {"kind": "silence", "sentence": "", "tts_text": ""}
    assert tts.max_allowed_duration(speech, args) == pytest.approx(6.5)
    assert tts.max_allowed_duration(control, args) == 30.0


def test_fingerprint_ignores_ephemeral_vllm_port(tmp_path):
    args = tts.parse_args(
        [
            "--sentences",
            str(tmp_path / "unused.jsonl"),
            "--out-dir",
            str(tmp_path / "out"),
            "--api-base",
            "http://127.0.0.1:18001",
        ]
    )
    record = {"id": "a", "sentence": "専門用語です", "tts_text": "せんもんようごです"}
    first = tts.synthesis_fingerprint(record, args)
    args.api_base = "http://127.0.0.1:28001"
    assert tts.synthesis_fingerprint(record, args) == first


def test_shard_selection_is_stable_and_disjoint():
    records = [{"id": str(index)} for index in range(10)]
    shards = [tts.select_shard(records, 4, index) for index in range(4)]
    assert [[row["id"] for row in shard] for shard in shards] == [
        ["0", "4", "8"],
        ["1", "5", "9"],
        ["2", "6"],
        ["3", "7"],
    ]
    assert {row["id"] for shard in shards for row in shard} == {str(index) for index in range(10)}


def test_authoritative_reading_is_derived_before_tts_when_tts_text_is_missing():
    record = {
        "id": "term-1",
        "kind": "term",
        "term": "活性汚泥法",
        "reading": "かっせいおでいほう",
        "sentence": "担当者が活性汚泥法の状態を確認します。",
    }

    assert tts.synthesis_text_for(record) == (
        "担当者がかっせいおでいほうの状態を確認します。"
    )


def test_input_does_not_stop_when_tts_text_disagrees_with_authoritative_reading(tmp_path):
    source = tmp_path / "sentences.jsonl"
    _write_jsonl(
        source,
        [
            {
                "id": "term-1",
                "kind": "term",
                "term": "活性汚泥法",
                "reading": "かっせいおでいほう",
                "sentence": "担当者が活性汚泥法の状態を確認します。",
                "tts_text": "担当者が活性汚泥方式の状態を確認します。",
            }
        ],
    )

    [record] = tts.load_sentences(source)
    assert tts.synthesis_text_for(record) == "担当者がかっせいおでいほうの状態を確認します。"


def test_pronunciation_transcript_accepts_term_or_authoritative_reading():
    record = {"term": "活性汚泥法", "reading": "かっせいおでいほう"}

    class Token:
        surface = "活性汚泥法"
        feature = SimpleNamespace(kana="カッセイオデイホウ")

    def tagger(_text):
        return [Token()]

    by_term = tts.evaluate_pronunciation_transcript(
        record, "活性汚泥法", tagger=tagger
    )
    by_reading = tts.evaluate_pronunciation_transcript(
        record, "担当者がカッセイオデイホウを確認しました。"
    )
    failed = tts.evaluate_pronunciation_transcript(
        record, "担当者が活性汚泥方式を確認しました。"
    )

    assert by_term["passed"] and by_term["match"] == "mora_exact"
    assert by_reading["passed"] and by_reading["match"] == "mora_exact"
    assert not failed["passed"]


def test_mora_check_tolerates_long_vowels_but_rejects_content_changes():
    record = {"term": "製法", "reading": "せいほう"}

    exact_long = tts.evaluate_pronunciation_transcript(record, "セーホー")
    shortened_long = tts.evaluate_pronunciation_transcript(record, "セホー")
    wrong_content = tts.evaluate_pronunciation_transcript(record, "セーソー")

    assert tts.kana_to_morae("せいほう") == tts.kana_to_morae("セーホー")
    assert exact_long["passed"] and exact_long["mora_distance"] == 0
    assert shortened_long["passed"]
    assert shortened_long["match"] == "mora_tolerant"
    assert not wrong_content["passed"]
    assert wrong_content["reason"] == "content_mora_mismatch"


def test_mora_check_rejects_empty_asr_and_missing_sokuon_or_hatsuon():
    empty = tts.evaluate_pronunciation_transcript(
        {"term": "製法", "reading": "せいほう"}, ""
    )
    missing_sokuon = tts.evaluate_pronunciation_transcript(
        {"term": "活性", "reading": "かっせい"}, "カセー"
    )
    missing_hatsuon = tts.evaluate_pronunciation_transcript(
        {"term": "安定", "reading": "あんてい"}, "アテー"
    )

    assert not empty["passed"] and empty["reason"] == "empty_asr"
    assert not missing_sokuon["passed"]
    assert missing_sokuon["content_edits"] >= 1
    assert not missing_hatsuon["passed"]
    assert missing_hatsuon["content_edits"] >= 1


def test_pronunciation_threshold_order_is_validated(tmp_path):
    with pytest.raises(SystemExit, match="0 <= pass <= uncertain <= 1"):
        tts.parse_args(
            [
                "--sentences",
                str(tmp_path / "unused.jsonl"),
                "--out-dir",
                str(tmp_path / "out"),
                "--pronunciation-pass-threshold",
                "0.4",
                "--pronunciation-uncertain-threshold",
                "0.2",
            ]
        )


def test_kanji_first_tts_keeps_passed_audio_and_falls_back_per_record(
    tmp_path, monkeypatch
):
    source = tmp_path / "sentences.jsonl"
    output = tmp_path / "audio"
    records = [
        {
            "id": "pass",
            "kind": "term",
            "term": "活性汚泥法",
            "reading": "かっせいおでいほう",
            "sentence": "活性汚泥法を確認します。",
            "tts_text": "かっせいおでいほうを確認します。",
        },
        {
            "id": "fallback",
            "kind": "term",
            "term": "最終沈殿池",
            "reading": "さいしゅうちんでんち",
            "sentence": "最終沈殿池を確認します。",
            "tts_text": "さいしゅうちんでんちを確認します。",
        },
    ]
    _write_jsonl(source, records)
    args = tts.parse_args(
        [
            "--sentences",
            str(source),
            "--out-dir",
            str(output),
            "--batch-size",
            "2",
            "--min-duration-sec",
            "0.01",
            "--pronunciation-check-model",
            "fake-ct2",
        ]
    )
    calls = []

    class Checker:
        def __init__(self, _args):
            pass

        def check(self, _audio, _sample_rate, rec):
            if rec["id"] == "pass":
                return {
                    "passed": True,
                    "transcript": rec["sentence"],
                    "match": "term",
                    "reason": "",
                }
            return {
                "passed": False,
                "transcript": "最終ちんでん地を確認します。",
                "match": "",
                "reason": "target_reading_not_recognized",
            }

    monkeypatch.setattr(tts, "PronunciationChecker", Checker)
    monkeypatch.setattr(tts, "load_model", lambda _args: object())

    def fake_backend(_model, _args, texts):
        calls.append(list(texts))
        return [(np.full(4_800, 0.1, dtype=np.float32), 24_000) for _ in texts]

    monkeypatch.setattr(tts, "_call_backend", fake_backend)
    tts.run(args)

    assert calls == [
        [records[0]["sentence"], records[1]["sentence"]],
        [records[1]["tts_text"]],
    ]
    manifest = {
        row["id"]: row
        for row in map(
            json.loads,
            (output / "manifest.jsonl").read_text(encoding="utf-8").splitlines(),
        )
    }
    assert manifest["pass"]["synthesis_text"] == records[0]["sentence"]
    assert manifest["pass"]["pronunciation_check"] == "passed"
    assert not manifest["pass"]["reading_fallback_used"]
    assert manifest["fallback"]["synthesis_text"] == records[1]["tts_text"]
    assert manifest["fallback"]["pronunciation_check"] == "fallback_reading"
    assert manifest["fallback"]["reading_fallback_used"]


def test_checker_load_failure_uses_reading_without_stopping(tmp_path, monkeypatch):
    source = tmp_path / "sentences.jsonl"
    output = tmp_path / "audio"
    record = {
        "id": "term",
        "kind": "term",
        "term": "活性汚泥法",
        "reading": "かっせいおでいほう",
        "sentence": "活性汚泥法を確認します。",
        "tts_text": "かっせいおでいほうを確認します。",
    }
    _write_jsonl(source, [record])
    args = tts.parse_args(
        [
            "--sentences",
            str(source),
            "--out-dir",
            str(output),
            "--min-duration-sec",
            "0.01",
            "--pronunciation-check-model",
            "missing-ct2",
        ]
    )
    monkeypatch.setattr(
        tts,
        "PronunciationChecker",
        lambda _args: (_ for _ in ()).throw(RuntimeError("not installed")),
    )
    monkeypatch.setattr(tts, "load_model", lambda _args: object())
    calls = []

    def fake_backend(_model, _args, texts):
        calls.append(list(texts))
        return [(np.full(4_800, 0.1, dtype=np.float32), 24_000)]

    monkeypatch.setattr(tts, "_call_backend", fake_backend)
    tts.run(args)

    assert calls == [[record["tts_text"]]]
    [manifest] = map(
        json.loads,
        (output / "manifest.jsonl").read_text(encoding="utf-8").splitlines(),
    )
    assert manifest["pronunciation_check"] == "checker_unavailable"
    assert manifest["reading_fallback_used"]
