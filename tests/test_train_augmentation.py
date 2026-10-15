import numpy as np

from scripts.train_whisper_lora import augment_waveform, build_training_rows


def sine(seconds: float = 0.5, sr: int = 16000) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype("float32")


def test_augmentation_is_deterministic_for_a_given_seed():
    wav = sine()
    a = augment_waveform(wav, 16000, 7, gain_db=6.0, noise_snr_db=15.0)
    b = augment_waveform(wav, 16000, 7, gain_db=6.0, noise_snr_db=15.0)
    c = augment_waveform(wav, 16000, 8, gain_db=6.0, noise_snr_db=15.0)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_disabled_augmentation_returns_the_original_audio():
    wav = sine()
    assert np.array_equal(augment_waveform(wav, 16000, 1), wav)


def test_noise_lowers_snr_without_clipping():
    wav = sine()
    noisy = augment_waveform(wav, 16000, 3, noise_snr_db=10.0)
    assert noisy.shape == wav.shape
    assert not np.array_equal(noisy, wav)
    assert float(np.max(np.abs(noisy))) <= 1.0


def test_gain_changes_level_but_stays_in_range():
    wav = sine()
    louder = augment_waveform(wav, 16000, 5, gain_db=6.0)
    assert float(np.max(np.abs(louder))) <= 1.0
    assert not np.allclose(louder, wav)


def test_build_training_rows_appends_augmented_copies():
    rows = [{"audio": "a.wav", "text": "あ"}, {"audio": "b.wav", "text": "い"}]
    assert build_training_rows(rows, 0) == [
        {"audio": "a.wav", "text": "あ", "aug_seed": -1},
        {"audio": "b.wav", "text": "い", "aug_seed": -1},
    ]

    expanded = build_training_rows(rows, 2)
    assert len(expanded) == 6
    assert sum(1 for row in expanded if row["aug_seed"] < 0) == 2
    seeds = [row["aug_seed"] for row in expanded if row["aug_seed"] >= 0]
    assert len(set(seeds)) == len(seeds)  # every copy gets its own augmentation
