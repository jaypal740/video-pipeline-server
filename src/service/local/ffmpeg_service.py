import json
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

    def probe(self, path: Path) -> dict:
        """Seconds, fps, width and height of a media file, from ffprobe."""
        result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                 "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(path)],
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr[-400:]}")
        info = json.loads(result.stdout)
        stream = (info.get("streams") or [{}])[0]
        num, _, den = (stream.get("r_frame_rate") or "0/1").partition("/")
        seconds = float(info.get("format", {}).get("duration") or 0)
        if seconds <= 0:
            raise RuntimeError(f"ffprobe read a zero-length file: {path}")
        return {"seconds": seconds, "fps": float(num) / float(den or 1), "width": stream.get("width"),
                "height": stream.get("height")}

    def run(self, args: list[str]) -> bytes:
        result = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", *args], capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.decode(errors='replace')[-400:]}")
        return result.stdout
