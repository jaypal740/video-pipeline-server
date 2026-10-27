import numpy as np
import whisperx

SAMPLE_RATE = 16000
MODEL_NAME = "WAV2VEC2_ASR_BASE_960H"


class ForcedAlignService:
    """Times known words against the audio with wav2vec2 (via WhisperX). The words are given, not
    heard, so it cannot mishear them; it only finds when each is said. The model loads on first use."""

    def __init__(self, device: str) -> None:
        self._device = device
        self._model = None
        self._meta = None

    @property
    def version(self) -> str:
        return f"whisperx-align:{MODEL_NAME}"

    def align(self, samples: np.ndarray, tokens: list[str]) -> list[tuple[float, float, float] | None]:
        """One (start, end, confidence) per token, in order; None for a token it could not time."""
        if self._model is None:
            self._model, self._meta = whisperx.load_align_model(language_code="en", device=self._device,
                                                                model_name=MODEL_NAME)
        segment = {"text": " ".join(tokens), "start": 0.0, "end": len(samples) / SAMPLE_RATE}
        result = whisperx.align([segment], self._model, self._meta, samples, self._device, return_char_alignments=False)
        words = [w for seg in result["segments"] for w in seg.get("words", [])]
        if len(words) != len(tokens):
            raise RuntimeError(f"aligner returned {len(words)} words for {len(tokens)} tokens")
        return [(float(w["start"]), float(w["end"]), float(w.get("score", 0.0))) if "start" in w else None
                for w in words]
