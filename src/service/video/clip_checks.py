import tempfile
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from src.service.local.ffmpeg_service import FfmpegService

# Checks that read every frame cost nothing and give the same answer every time, so they run before the
# judge. Frames are read once, small and grey. Motion is measured where the clip changes most: each pixel's
# mean change between consecutive sampled frames (0-255), and the score is the busiest 1% of pixels, so a
# small action in a wide shot counts and a still clip with a faint shimmer does not. The limits come from
# the 2026-11-02 rough cut: real actions scored 13.6 and up, a near-still clip 9.3, frozen endings below 2.
SAMPLE_FPS = 4
WIDTH, HEIGHT = 160, 90
BUSIEST = 99                # percentile of per-pixel change that stands for the clip's motion
MIN_MOTION = 10.0           # below this over the whole clip, it reads as a still
STILL = 3.0                 # a last second changing less than this is frozen
# A clip drawn to an end frame often reaches it early and holds it. That held tail is cut off at assembly
# and the rest stretched over the slot, slowed at most MAX_SLOW (assembly's limit too), so the clip passes
# when what is left to hold on its ending is at most MAX_HOLD: a beat on a finished action reads as natural,
# a second does not (2026-11-02, s05: a 0.25 s hold passes, 1.2 s fails).
MAX_SLOW = 1.25
MAX_HOLD = 0.5
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
    active_seconds: float | None = None         # where the picture stops moving, when it holds still after


def motion_score(grey: np.ndarray) -> float:
    """grey: (n, h, w) uint8; how much the busiest part of the picture changes between consecutive frames."""
    if len(grey) < 2:
        return 0.0
    frames = grey.astype(np.float32)
    return float(np.percentile(np.abs(frames[1:] - frames[:-1]).mean(axis=0), BUSIEST))


def tail_motion(grey: np.ndarray, fps: float, seconds: float = 1.0) -> float:
    tail = grey[-max(2, int(round(fps * seconds))):]
    return motion_score(tail)


def active_seconds(grey: np.ndarray, fps: float) -> float:
    """How long the picture keeps moving: up to the last sampled step whose busiest part changes by STILL
    or more. A clip that moves to its end gives its whole length."""
    if len(grey) < 2:
        return 0.0
    frames = grey.astype(np.float32)
    steps = np.percentile(np.abs(frames[1:] - frames[:-1]).reshape(len(frames) - 1, -1), BUSIEST, axis=1)
    moving = np.nonzero(steps >= STILL)[0]
    return float((moving[-1] + 1) / fps) if len(moving) else 0.0


def black_frames(grey: np.ndarray) -> int:
    return int((grey.reshape(len(grey), -1).max(axis=1) < BLACK).sum())


def problems_of(checks: "ClipChecks", wanted_seconds: float) -> list[str]:
    found = []
    if checks.seconds < wanted_seconds - DURATION_SLACK:
        found.append(f"the clip is {checks.seconds:.1f}s, {wanted_seconds}s was asked for")
    if checks.motion < MIN_MOTION:
        found.append(f"almost nothing moves (motion {checks.motion:.2f}); the picture reads as a still")
    elif checks.tail_motion < STILL and (checks.active_seconds or 0.0) * MAX_SLOW < wanted_seconds - MAX_HOLD:
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
        if checks.tail_motion < STILL:                          # it holds still at the end: from where
            checks.active_seconds = round(min(active_seconds(grey, SAMPLE_FPS), checks.seconds), 2)
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
