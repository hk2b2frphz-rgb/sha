from scripts.synthesize_speech_kokoro import build_manifest_entry


def test_kokoro_manifest_keeps_original_asr_target():
    entry = build_manifest_entry(
        {
            "id": "x",
            "term": "活性汚泥法",
            "sentence": "活性汚泥法を確認します。",
            "tts_text": "かっせいおでいほうを確認します。",
        },
        "かっせいおでいほうを確認します。",
        "wav/x.wav",
        2.5,
        "jf_alpha",
    )

    assert entry["sentence"] == "活性汚泥法を確認します。"
    assert entry["tts_text"] == "かっせいおでいほうを確認します。"
    assert entry["synthesis_text"] == "かっせいおでいほうを確認します。"
