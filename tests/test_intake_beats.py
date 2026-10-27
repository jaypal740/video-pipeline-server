import numpy as np

from src.service.intake.intake_beats import beat_strength, bpm_of, onset_envelope

SR = 16000


def clicks(bpm: float, seconds: float = 20.0) -> np.ndarray:
    y = np.zeros(int(seconds * SR), dtype=np.float32)
    for t in np.arange(0.5, seconds, 60 / bpm):
        i = int(t * SR)
        y[i:i + 200] = np.random.default_rng(int(t * 1000)).uniform(-1, 1, 200)
    return y


def test_bpm_from_beat_spacing():
    assert bpm_of([0.0, 0.5, 1.0, 1.5]) == 120.0
    assert bpm_of([1.0]) is None


def test_a_steady_click_track_has_a_strong_beat():
    assert beat_strength(onset_envelope(clicks(120), SR), 120.0) > 0.5


def test_noise_has_no_beat():
    noise = np.random.default_rng(0).normal(0, 0.1, 20 * SR).astype(np.float32)
    assert beat_strength(onset_envelope(noise, SR), 120.0) < 0.1


def test_no_bpm_means_no_strength():
    assert beat_strength(onset_envelope(clicks(120), SR), None) == 0.0
