import json
from pathlib import Path

from src.schema.song.song_analysis_schema import LyricLine, TimedWord
from src.service.intake.intake_lyrics import parse_lyrics

# A song's JamWise export, kept next to its audio.
JAMWISE_FILE = "jamwise.json"
# A line shorter than this cannot carry a shot; JamWise has timed one at 0.28 s.
UNCLEAR_SHORTER = 0.5
# How far past the end of the audio a timing may run before it is wrong.
END_SLACK = 0.5


def read_export(path: Path) -> dict:
    """The result part of a JamWise job export: lyrics, lineTiming and the rest."""
    result = json.loads(Path(path).read_text()).get("result") or {}
    if not result.get("lyrics") or not result.get("lineTiming"):
        raise ValueError(f"{path} is not a JamWise export with lyrics and lineTiming")
    return result


def jamwise_lines(timing: list[dict], lyrics: list[str], duration: float) -> tuple[list[TimedWord], list[LyricLine]]:
    """JamWise's own line and word timing as our timed lyrics, checked against the lyrics and the audio."""
    if len(timing) != len(lyrics):
        raise ValueError(f"JamWise timed {len(timing)} lines, the lyrics have {len(lyrics)}")
    words, lines = [], []
    for li, (text, line) in enumerate(zip(lyrics, timing)):
        written, timed = text.split(), line["words"]
        if [l for _, l in parse_lyrics(line["text"])] != [text] or len(timed) != len(written):
            raise ValueError(f"line {li} {text!r} does not match JamWise's {line['text']!r}")
        start, end = line["startTime"], line["endTime"]
        if not 0 <= start <= end <= duration + END_SLACK or any(w["end"] < w["start"] for w in timed):
            raise ValueError(f"line {li} {text!r} has impossible timing {start}-{end}s in {duration}s of audio")
        for word, w in zip(written, timed):
            words.append(TimedWord(index=len(words), line_index=li, text=word, start=w["start"], end=w["end"],
                                   confidence=1.0))
        lines.append(LyricLine(index=li, text=text, start=start, end=end, confidence=1.0,
                               unclear=end - start < UNCLEAR_SHORTER))
    return words, lines
