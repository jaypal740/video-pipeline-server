import subprocess
from pathlib import Path

from src.schema.song.song_analysis_schema import LyricLine, SongAnalysis, TimedWord
from src.service.intake.preview_service import PreviewService
from src.service.local.ffmpeg_service import FfmpegService


def test_check_video_covers_the_song(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)          # a relative output folder, as the CLI passes
    audio = tmp_path / "a.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=300:duration=3", str(audio)], check=True)
    analysis = SongAnalysis(
        cache_key="k", audio_sha256="a", lyrics_sha256="l", duration_s=3.0, accepted=True,
        words=[TimedWord(index=0, line_index=0, text="hello", start=0.5, end=1.0, confidence=0.9),
               TimedWord(index=1, line_index=0, text="world", start=1.0, end=1.6, confidence=0.9)],
        lines=[LyricLine(index=0, text="hello world", start=0.5, end=1.6, confidence=0.9)],
        bpm=120.0, beats=[0.5, 1.0, 1.5], downbeats=[0.5])
    out = PreviewService(FfmpegService()).render(analysis, audio, Path("out"))
    seconds = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
                                   capture_output=True, text=True).stdout)
    assert abs(seconds - 3.0) < 0.2
