import numpy as np
from beat_this.inference import Audio2Beats

SAMPLE_RATE = 22050
CHECKPOINT = "final0"


class BeatService:
    """Beats and downbeats with beat_this. The model loads on first use."""

    def __init__(self, device: str) -> None:
        self._device = device
        self._model: Audio2Beats | None = None

    @property
    def version(self) -> str:
        return f"beat_this:{CHECKPOINT}"

    def track(self, samples: np.ndarray) -> tuple[list[float], list[float]]:
        if self._model is None:
            self._model = Audio2Beats(checkpoint_path=CHECKPOINT, device=self._device, dbn=False)
        beats, downbeats = self._model(samples, SAMPLE_RATE)
        return [round(float(b), 3) for b in beats], [round(float(d), 3) for d in downbeats]
