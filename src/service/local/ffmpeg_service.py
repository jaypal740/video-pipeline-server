import subprocess
from pathlib import Path

import numpy as np


class FfmpegService:
    """Every ffmpeg call. Audio is always decoded here, so no library's own file loader is relied on
    (torchaudio's failed on plain mp3 and wav)."""

    def decode(self, audio: Path, sample_rate: int) -> np.ndarray:
        """Mono float32 samples at sample_rate."""
        raw = self.run(["-i", str(audio), "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-"])
        return np.frombuffer(raw, dtype=np.float32)

    def run(self, args: list[str]) -> bytes:
        result = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", *args], capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.decode(errors='replace')[-400:]}")
        return result.stdout
