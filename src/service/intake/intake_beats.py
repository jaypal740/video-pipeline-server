import numpy as np

FRAME_HOP = 160          # 10 ms frames at 16 kHz
FFT_SIZE = 1024
# Below this the music has no steady pulse (narration beds scored 0.02-0.04), and cuts follow the voice.
# Not yet measured on a song with a real beat; set from the first one.
BEATS_USABLE_MIN = 0.2


def onset_envelope(samples: np.ndarray, sr: int) -> np.ndarray:
    """Onset strength per 10 ms frame: summed positive change in log magnitude, 30 Hz - 8 kHz."""
    hop = FRAME_HOP * sr // 16000
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(samples, (FFT_SIZE // 2, FFT_SIZE // 2)), FFT_SIZE)[::hop]
    spec = np.abs(np.fft.rfft(frames * np.hanning(FFT_SIZE), axis=1))
    freqs = np.fft.rfftfreq(FFT_SIZE, 1 / sr)
    mag = np.log1p(spec[:, (freqs >= 30) & (freqs < 8000)])
    return np.maximum(np.diff(mag, axis=0, prepend=mag[:1]), 0).sum(axis=1)


def bpm_of(beats: list[float]) -> float | None:
    if len(beats) < 2:
        return None
    return round(60.0 / float(np.median(np.diff(beats))), 1)


def beat_strength(envelope: np.ndarray, bpm: float | None, frames_per_second: float = 100.0) -> float:
    """How strongly the music repeats at its beat period: autocorrelation of onset strength at one beat.
    Near 0 for a narration bed with no pulse; a song with a real beat repeats strongly."""
    if bpm is None or len(envelope) < 4:
        return 0.0
    env = envelope - envelope.mean()
    ac = np.correlate(env, env, "full")[len(env) - 1:]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    lag = int(round(60.0 / bpm * frames_per_second))
    if lag + 3 > len(ac):
        return 0.0
    return round(float(max(ac[max(lag - 2, 1):lag + 3].max(), 0.0)), 3)
