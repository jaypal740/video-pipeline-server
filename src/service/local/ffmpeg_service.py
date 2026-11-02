import json
import re
import subprocess
from pathlib import Path

import numpy as np


class FfmpegService:
    """Every ffmpeg call. Audio is always decoded here, so no library's own file loader is relied on
    (torchaudio's failed on plain mp3 and wav). A call that runs past timeout_s is stopped and fails, so a
    hung ffmpeg never stalls a stage."""

    def __init__(self, timeout_s: float = 600.0) -> None:
        self._timeout_s = timeout_s

    def decode(self, audio: Path, sample_rate: int) -> np.ndarray:
        """Mono float32 samples at sample_rate."""
        raw = self.run(["-i", str(audio), "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-"])
        return np.frombuffer(raw, dtype=np.float32)

    def probe(self, path: Path) -> dict:
        """Seconds, fps, width and height of a media file, from ffprobe."""
        result = self._exec(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                             "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(path)], text=True)
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

    def packets(self, path: Path) -> dict[str, int]:
        """Packets per stream type ({"video": 1440, "audio": 2813}), counted without decoding; a video
        stream's packets are its frames."""
        result = self._exec(["ffprobe", "-v", "error", "-count_packets", "-show_entries",
                             "stream=codec_type,nb_read_packets", "-of", "json", str(path)], text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr[-400:]}")
        counts: dict[str, int] = {}
        for stream in json.loads(result.stdout).get("streams", []):
            kind = stream.get("codec_type", "")
            counts[kind] = counts.get(kind, 0) + int(stream.get("nb_read_packets") or 0)
        return counts

    def black(self, path: Path, seconds: float, pixel: float, picture: float) -> list[tuple[float, float]]:
        """(start, end) of every stretch of at least `seconds` where `picture` of the pixels are darker than
        `pixel`, from ffmpeg's blackdetect (which reports on the info log, so -v error is not used here)."""
        result = self._exec(["ffmpeg", "-v", "info", "-nostdin", "-i", str(path), "-an", "-vf",
                             f"blackdetect=d={seconds}:pix_th={pixel}:pic_th={picture}", "-f", "null", "-"])
        log = result.stderr.decode(errors="replace")
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {log[-400:]}")
        return [(float(a), float(b)) for a, b in re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", log)]

    def run(self, args: list[str]) -> bytes:
        result = self._exec(["ffmpeg", "-v", "error", "-nostdin", "-y", *args])
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.decode(errors='replace')[-400:]}")
        return result.stdout

    def _exec(self, command: list[str], text: bool = False) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(command, capture_output=True, text=text, timeout=self._timeout_s)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"{command[0]} did not finish in {self._timeout_s:g}s and was stopped: "
                               f"{' '.join(command)[:300]}") from None
