import numpy as np

from scripts.synthesize_speech import add_lead_silence as add_qwen_lead_silence
from scripts.synthesize_speech import prepare_tts_text as prepare_qwen_text
from scripts.synthesize_speech_kokoro import add_lead_silence as add_kokoro_lead_silence
from scripts.synthesize_speech_kokoro import prepare_tts_text as prepare_kokoro_text


def test_leading_silence_padding_uses_requested_duration():
    audio = np.array([0.25, -0.5], dtype=np.float32)
    for add_silence in (add_qwen_lead_silence, add_kokoro_lead_silence):
        padded = add_silence(audio, 16_000, 250)
        assert padded.shape == (4_002,)
        assert np.all(padded[:4_000] == 0)
        assert np.array_equal(padded[4_000:], audio)


def test_zero_leading_silence_keeps_audio():
    audio = np.array([0.25], dtype=np.float32)
    assert np.array_equal(add_kokoro_lead_silence(audio, 24_000, 0), audio)


def test_prepend_pause_only_changes_text_sent_to_tts():
    for prepare_text in (prepare_qwen_text, prepare_kokoro_text):
        assert prepare_text("原文です", True) == "、原文です"
        assert prepare_text("原文です", False) == "原文です"
