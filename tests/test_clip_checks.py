import shutil
import subprocess

import numpy as np
import pytest

from src.service.local.ffmpeg_service import FfmpegService
from src.service.video.clip_checks import (HEIGHT, WIDTH, ClipCheckService, ClipChecks, black_frames, motion_score,
                                           problems_of, tail_motion)


def clip(frames: int, moving: bool, freeze_last: int = 0, black: int = 0) -> np.ndarray:
    """A synthetic grey clip: a bright square that slides right, or stays put."""
    grey = np.zeros((frames, HEIGHT, WIDTH), dtype=np.uint8) + 60
    for i in range(frames):
        x = (i * 6 if moving else 20) if i < frames - freeze_last else ((frames - freeze_last - 1) * 6 if moving else 20)
        grey[i, 30:60, x:x + 30] = 220
    for i in range(black):
        grey[i] = 0
    return grey


def test_motion_freeze_and_black_metrics_on_synthetic_frames():
    assert motion_score(clip(16, moving=True)) > 1.0
    assert motion_score(clip(16, moving=False)) == 0.0
    assert tail_motion(clip(16, moving=True, freeze_last=6), fps=4) == 0.0
    assert tail_motion(clip(16, moving=True), fps=4) > 1.0
    assert black_frames(clip(16, moving=True, black=2)) == 2
    night = clip(16, moving=True)                                # a dark sky with one bright spot is not black
    night[:, :, :] = 4
    night[:, 10:14, 20:24] = 255
    assert black_frames(night) == 0


def test_problems_name_what_the_code_saw():
    ok = ClipChecks(seconds=5.0, fps=24, width=1280, height=720, motion=3.0, tail_motion=2.0, black_frames=0, problems=[])
    assert problems_of(ok, 5) == []
    still = ok.model_copy(update={"motion": 0.2})
    assert problems_of(still, 5) == ["almost nothing moves (motion 0.20); the picture reads as a still"]
    frozen = ok.model_copy(update={"tail_motion": 0.0})
    assert problems_of(frozen, 5) == ["the picture freezes before the end; the last second is a still frame"]
    short = ok.model_copy(update={"seconds": 3.9, "black_frames": 1})
    assert problems_of(short, 5) == ["the clip is 3.9s, 5s was asked for", "1 black frames"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs the ffmpeg binary")
def test_a_real_clip_is_read_by_ffmpeg(tmp_path):
    def lavfi(source: str) -> bytes:
        out = tmp_path / f"{abs(hash(source))}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", source, "-t", "2", "-r", "24",
                        "-pix_fmt", "yuv420p", str(out)], check=True)
        return out.read_bytes()

    service = ClipCheckService(FfmpegService())
    moving = service.check(lavfi("testsrc=size=320x180:rate=24"), wanted_seconds=2)
    assert moving.problems == [] and moving.seconds == pytest.approx(2.0, abs=0.1) and moving.width == 320
    still = service.check(lavfi("color=c=blue:size=320x180:rate=24"), wanted_seconds=2)
    assert any("almost nothing moves" in p for p in still.problems)
    frames = service.frames(moving.model_dump() and lavfi("testsrc=size=320x180:rate=24"), moving.seconds)
    assert len(frames) == 3 and all(f.startswith(b"\x89PNG") for f in frames)
