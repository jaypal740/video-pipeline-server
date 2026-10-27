import textwrap
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.schema.song.song_analysis_schema import LyricLine, SongAnalysis, TimedWord
from src.service.intake.intake_beats import BEATS_USABLE_MIN
from src.service.local.ffmpeg_service import FfmpegService

WIDTH, HEIGHT = 1280, 720
TICK_RATE = 44100
BACKGROUND, TEXT, SPOKEN, DIM, WARN = (18, 20, 28), (120, 126, 140), (255, 214, 92), (110, 118, 136), (235, 110, 100)


class PreviewService:
    """The check video for a person to judge intake by ear: each line on screen while it is said, the
    word being said highlighted, a low tick on each beat and a high tick on each downbeat."""

    def __init__(self, ffmpeg: FfmpegService) -> None:
        self._ffmpeg = ffmpeg
        self._font = ImageFont.load_default(size=44)
        self._small = ImageFont.load_default(size=24)

    def render(self, analysis: SongAnalysis, audio: Path, out_dir: Path) -> Path:
        out_dir = out_dir.resolve()          # ffmpeg reads paths inside the concat list relative to the list
        out_dir.mkdir(parents=True, exist_ok=True)
        cards = out_dir / "cards"
        cards.mkdir(exist_ok=True)
        events = self._timeline(analysis)
        listing = []
        for i, (start, end, line, word) in enumerate(events):
            path = cards / f"{i:05d}.png"
            self._card(path, analysis, line, word)
            listing.append(f"file '{path}'\nduration {end - start:.3f}\n")
        concat = out_dir / "cards.txt"
        concat.write_text("".join(listing) + f"file '{cards / f'{len(events) - 1:05d}.png'}'\n")
        ticks = out_dir / "ticks.wav"
        self._tick_track(ticks, analysis)
        out = out_dir / "check.mp4"
        self._ffmpeg.run(["-f", "concat", "-safe", "0", "-i", str(concat), "-i", str(audio), "-i", str(ticks),
                          "-filter_complex", "[1:a][2:a]amix=inputs=2:duration=first:normalize=0[a]", "-map", "0:v",
                          "-map", "[a]", "-t", f"{analysis.duration_s:.3f}", "-r", "25", "-pix_fmt", "yuv420p",
                          "-c:v", "libx264", "-c:a", "aac", str(out)])
        return out

    def _timeline(self, analysis: SongAnalysis) -> list[tuple[float, float, LyricLine | None, TimedWord | None]]:
        """Cut the song at every word start and end; each piece shows its line and the word being said."""
        cuts = sorted({0.0, analysis.duration_s, *(t for w in analysis.words for t in (w.start, w.end))})
        events = []
        for start, end in zip(cuts, cuts[1:]):
            if end - start < 0.001:
                continue
            mid = (start + end) / 2
            line = next((l for l in analysis.lines if l.start <= mid < l.end), None)
            word = next((w for w in analysis.words if w.start <= mid < w.end), None)
            events.append((start, end, line, word))
        return events

    def _card(self, path: Path, analysis: SongAnalysis, line: LyricLine | None, word: TimedWord | None) -> None:
        img = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
        draw = ImageDraw.Draw(img)
        if line is not None:
            own = [w for w in analysis.words if w.line_index == line.index]
            rows = textwrap.wrap(" ".join(w.text for w in own), 40)
            y, k = HEIGHT // 2 - len(rows) * 30, 0
            for row in rows:
                n = len(row.split())
                parts = own[k:k + n]
                k += n
                x = (WIDTH - draw.textlength(row, font=self._font)) / 2
                for w in parts:
                    draw.text((x, y), w.text, font=self._font, fill=SPOKEN if word is w else TEXT)
                    x += draw.textlength(w.text + " ", font=self._font)
                y += 60
            label = f"line {line.index}  {line.start:.2f}-{line.end:.2f}s  confidence {line.confidence:.2f}"
            draw.text((24, HEIGHT - 44), label + ("   UNCLEAR: check this line" if line.unclear else ""),
                      font=self._small, fill=WARN if line.unclear else DIM)
        beats = "beats usable" if analysis.beat_strength >= BEATS_USABLE_MIN else "no steady beat: cuts follow the voice"
        draw.text((24, 20), f"{analysis.bpm or 0:.0f} BPM  beat strength {analysis.beat_strength:.2f}  ({beats})",
                  font=self._small, fill=DIM)
        img.save(path)

    def _tick_track(self, path: Path, analysis: SongAnalysis) -> None:
        track = np.zeros(int((analysis.duration_s + 1) * TICK_RATE), dtype=np.float32)
        downs = set(analysis.downbeats)
        for b in analysis.beats:
            hz = 1800 if b in downs else 900
            t = np.arange(int(0.035 * TICK_RATE)) / TICK_RATE
            burst = 0.3 * np.sin(2 * np.pi * hz * t) * np.hanning(len(t))
            i = int(b * TICK_RATE)
            track[i:i + len(burst)] += burst[: max(len(track) - i, 0)]
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(TICK_RATE)
            w.writeframes((np.clip(track, -1, 1) * 32767).astype(np.int16).tobytes())
