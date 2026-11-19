import subprocess

import pytest

from src.service.local.ffmpeg_service import FfmpegService


def test_a_feed_that_is_interrupted_kills_ffmpeg_and_reaps_it(monkeypatch):
    started = []
    real = subprocess.Popen
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: started.append(real(*a, **k)) or started[-1])

    def frames():
        yield b"\x00" * 12
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        FfmpegService().feed(["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "2x2", "-r", "1", "-i", "-", "-f", "null", "-"],
                             frames())
    assert started[0].returncode is not None                    # waited for, not left a zombie
