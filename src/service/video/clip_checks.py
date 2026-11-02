import tempfile
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from src.service.local.ffmpeg_service import FfmpegService

# Checks that read every frame cost nothing and give the same answer every time, so they run before the
# judge. Frames are read once, small and grey; the numbers are first guesses to be set on the bake-off
# clips. Motion is the mean absolute change between consecutive sampled frames on a 0-255 scale.
SAMPLE_FPS = 4
WIDTH, HEIGHT = 160, 90
MIN_MOTION = 1.0            # below this over the whole clip, it reads as a still
STILL = 0.3                 # a last second changing less than this is frozen
BLACK = 32                  # a frame with no pixel brighter than this is black; a night sky with stars is not
DURATION_SLACK = 0.5        # seconds short of the requested length that still pass


class ClipChecks(BaseModel):
    seconds: float
    fps: float
    width: int | None
    height: int | None
    motion: float
    tail_motion: float
    black_frames: int
    problems: list[str]


def motion_score(grey: np.ndarray) -> float:
    """grey: (n, h, w) uint8; the mean change between consecutive frames, averaged over the clip."""
    if len(grey) < 2:
        return 0.0
    frames = grey.astype(np.float32)
    return float(np.abs(frames[1:] - frames[:-1]).mean())


def tail_motion(grey: np.ndarray, fps: float, seconds: float = 1.0) -> float:
    tail = grey[-max(2, int(round(fps * seconds))):]
    return motion_score(tail)


def black_frames(grey: np.ndarray) -> int:
    return int((grey.reshape(len(grey), -1).max(axis=1) < BLACK).sum())


def problems_of(checks: "ClipChecks", wanted_seconds: float) -> list[str]:
    found = []
    if checks.seconds < wanted_seconds - DURATION_SLACK:
        found.append(f"the clip is {checks.seconds:.1f}s, {wanted_seconds}s was asked for")
    if checks.motion < MIN_MOTION:
        found.append(f"almost nothing moves (motion {checks.motion:.2f}); the picture reads as a still")
    elif checks.tail_motion < STILL:
        found.append("the picture freezes before the end; the last second is a still frame")
    if checks.black_frames:
        found.append(f"{checks.black_frames} black frames")
    return found


class ClipCheckService:
    """Reads a clip's frames with ffmpeg: the metrics for the code checks and three full frames (first,
    middle, last) as PNG for the judge."""

    def __init__(self, ffmpeg: FfmpegService) -> None:
        self._ffmpeg = ffmpeg

    def check(self, data: bytes, wanted_seconds: float) -> ClipChecks:
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            f.write(data)
            f.flush()
            path = Path(f.name)
            info = self._ffmpeg.probe(path)
            raw = self._ffmpeg.run(["-i", str(path), "-vf", f"fps={SAMPLE_FPS},scale={WIDTH}:{HEIGHT}", "-f", "rawvideo",
                                    "-pix_fmt", "gray", "-"])
        grey = np.frombuffer(raw, dtype=np.uint8).reshape(-1, HEIGHT, WIDTH)
        checks = ClipChecks(seconds=info["seconds"], fps=info["fps"], width=info["width"], height=info["height"],
                            motion=round(motion_score(grey), 2), tail_motion=round(tail_motion(grey, SAMPLE_FPS), 2),
                            black_frames=black_frames(grey), problems=[])
        checks.problems = problems_of(checks, wanted_seconds)
        return checks

    def frames(self, data: bytes, seconds: float) -> list[bytes]:
        """The first, middle and last frame as PNG bytes."""
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            f.write(data)
            f.flush()
            return [self._ffmpeg.run(["-ss", f"{t:.3f}", "-i", f.name, "-frames:v", "1", "-f", "image2pipe",
                                      "-vcodec", "png", "-"])
                    for t in (0.0, seconds / 2, max(seconds - 0.15, 0.0))]
