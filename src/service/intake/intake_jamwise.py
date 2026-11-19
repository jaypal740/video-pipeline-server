import json
from pathlib import Path

from src.schema.song.song_analysis_schema import LyricLine, TimedWord
from src.service.intake.intake_lyrics import parse_lyrics

# A song's JamWise export, kept next to its audio.
JAMWISE_FILE = "jamwise.json"
# How far past the end of the audio a timing may run before it is wrong.
END_SLACK = 0.5


def read_export(path: Path) -> dict:
    """The result part of a JamWise job export: lyrics, lineTiming and the rest."""
    result = json.loads(Path(path).read_text()).get("result") or {}
    if not result.get("lyrics") or not result.get("lineTiming"):
        raise ValueError(f"{path} is not a JamWise export with lyrics and lineTiming")
    return result


def export_beside(audio_path: str) -> dict | None:
    """The JamWise export kept next to a song's audio, if the song came from JamWise."""
    export = Path(audio_path).parent / JAMWISE_FILE
    return read_export(export) if export.is_file() else None


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_timing(timing: list[dict]) -> None:
    """Every line and word has the numbers it needs, so a bad export is named, not a bare KeyError."""
    for li, line in enumerate(timing):
        for key in ("startTime", "endTime"):
            if not _number(line.get(key)):
                raise ValueError(f"line {li} of JamWise's lineTiming has no numeric {key}")
        if not isinstance(line.get("words"), list):
            raise ValueError(f"line {li} of JamWise's lineTiming has no words list")
        for wi, w in enumerate(line["words"]):
            for key in ("start", "end"):
                if not _number(w.get(key)):
                    raise ValueError(f"line {li} word {wi} of JamWise's lineTiming has no numeric {key}")
        if line["endTime"] < line["startTime"]:
            raise ValueError(f"line {li} of JamWise's lineTiming ends before it starts")


def jamwise_lines(timing: list[dict], lyrics: list[str], duration: float) -> tuple[list[TimedWord], list[LyricLine]]:
    """JamWise's own line and word timing as our timed lyrics, checked against the lyrics and the audio."""
    if len(timing) != len(lyrics):
        raise ValueError(f"JamWise timed {len(timing)} lines, the lyrics have {len(lyrics)}")
    _check_timing(timing)
    words, lines = [], []
    for li, (text, line) in enumerate(zip(lyrics, timing)):
        written, timed = text.split(), line["words"]
        if [l for _, l in parse_lyrics(line["text"])] != [text] or len(timed) != len(written):
            raise ValueError(f"line {li} {text!r} does not match JamWise's {line['text']!r}")
        starts = [w["start"] for w in timed]
        start, end = min(line["startTime"], *starts), line["endTime"]      # JamWise stamps some first words, and their lines, after the second word
        if not 0 <= start <= end <= duration + END_SLACK or any(w["end"] < w["start"] for w in timed):
            raise ValueError(f"line {li} {text!r} has impossible timing {start}-{end}s in {duration}s of audio")
        for i, (word, w) in enumerate(zip(written, timed)):
            at = min(starts[i:])                                           # a word is never sung after the words that follow it
            words.append(TimedWord(index=len(words), line_index=li, text=word, start=at,
                                   end=w["end"] if at == w["start"] else at, confidence=1.0))
        lines.append(LyricLine(index=li, text=text, start=start, end=end, confidence=1.0))   # JamWise sang it: never unclear
    return words, lines
