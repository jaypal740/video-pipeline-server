import io
import tempfile
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image
from pydantic import BaseModel

from src.service.drawing.cutout import cut
from src.service.local.ffmpeg_service import FfmpegService

# How much a frame differs from the loop's first frame is measured where it differs most: each pixel's mean
# change (0-255), then the busiest 1% of pixels, so a small subject counts as much as a large one. On the
# 2026-11-11 trial loop (a caterpillar swaying) the frames in the middle scored 160-170, the last frame 11,
# and the second frame 6 from compression alone.
BUSIEST = 99
CLOSES = 30.0               # a last frame further than this from the first makes the loop jump when it repeats
MOVES = 40.0                # a loop no frame of which gets this far from the first reads as a still picture


class LoopCheck(BaseModel):
    """What code read off a loop. gap: how far the last frame is from the first; motion: how far the
    furthest frame is; uncut: what made frames unusable as cut-outs, one line per such frame."""

    seconds: float
    fps: float
    width: int
    height: int
    frames: int
    gap: float
    motion: float
    uncut: list[str]
    problems: list[str] = []


def cut_frames(frames: np.ndarray, sink: BinaryIO) -> tuple[float, float, int, list[str]]:
    """Cut every frame (n, height, width, 3) out and write it to `sink` as raw RGBA. Returns the gap, the
    motion, the index of the frame furthest from the first (by its whole picture, which tells two frames
    apart when their busiest parts have both changed completely), and the frames that are no cut-outs."""
    first, far, whole, uncut = frames[0].astype(np.float32), [], [], []
    for frame in frames:
        rgba, found = cut(frame)
        sink.write(rgba.tobytes())
        uncut += found[:1]
        change = np.abs(frame.astype(np.float32) - first).mean(axis=2)
        far.append(float(np.percentile(change, BUSIEST)))
        whole.append(float(change.mean()))
    return far[-1], max(far), int(np.argmax(whole)), uncut


def problems_of(check: LoopCheck) -> list[str]:
    found = []
    if check.uncut:
        found.append(f"{len(check.uncut)} of {check.frames} frames cannot be cut out: {check.uncut[0]}")
    if check.gap > CLOSES:
        found.append(f"the last frame is not the first again (differs by {check.gap:.0f}); the loop would jump "
                     "each time it repeats")
    if check.motion < MOVES:
        found.append(f"almost nothing moves (motion {check.motion:.0f}); the loop reads as a still picture")
    return found


class LoopCutService:
    """Reads a loop's frames with ffmpeg, cuts each one out and checks the loop by code. Gives the loop as a
    transparent WebM (none when a frame is no cut-out) and the frame furthest from the first, as PNG, for
    the judge."""

    def __init__(self, ffmpeg: FfmpegService) -> None:
        self._ffmpeg = ffmpeg

    def cut(self, data: bytes) -> tuple[LoopCheck, bytes | None, bytes]:
        with tempfile.TemporaryDirectory() as folder:
            clip, raw, out = Path(folder, "loop.mp4"), Path(folder, "cut.rgba"), Path(folder, "cut.webm")
            clip.write_bytes(data)
            info = self._ffmpeg.probe(clip)
            width, height = info["width"], info["height"]
            frames = np.frombuffer(self._ffmpeg.run(["-i", str(clip), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]),
                                   dtype=np.uint8).reshape(-1, height, width, 3)
            with raw.open("wb") as sink:
                gap, motion, furthest, uncut = cut_frames(frames, sink)
            check = LoopCheck(seconds=info["seconds"], fps=info["fps"], width=width, height=height, frames=len(frames),
                              gap=round(gap, 1), motion=round(motion, 1), uncut=uncut)
            check.problems = problems_of(check)
            if not uncut:
                self._ffmpeg.run(["-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{width}x{height}", "-r", f"{info['fps']:g}",
                                  "-i", str(raw), "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "24",
                                  "-auto-alt-ref", "0", str(out)])
            picture = io.BytesIO()
            Image.fromarray(frames[furthest]).save(picture, format="PNG")
            return check, None if uncut else out.read_bytes(), picture.getvalue()
