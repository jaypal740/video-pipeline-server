from src.schema.song.song_analysis_schema import LyricLine, TimedWord
from src.service.intake.intake_lyrics import spoken

# A line the aligner is less sure of than this may not be what the audio says. Real lines across
# 11 test songs scored 0.57 and up; two deliberately wrong lines scored 0.17 and 0.46.
UNCLEAR_BELOW = 0.5
# A song whose lines average below this is not the audio of these lyrics.
REJECT_BELOW = 0.5


def spoken_tokens(lines: list[str]) -> tuple[list[str], list[tuple[int, int]]]:
    """Every token to align, and for each token which (line, word) of the written lyrics it came from."""
    tokens, owners = [], []
    for li, line in enumerate(lines):
        for wi, word in enumerate(line.split()):
            for tok in spoken(word):
                tokens.append(tok)
                owners.append((li, wi))
    return tokens, owners


def timed_words(lines: list[str], owners: list[tuple[int, int]],
                aligned: list[tuple[float, float, float] | None]) -> list[TimedWord]:
    """Fold aligned tokens back onto the written words. A word with no timed token (punctuation only)
    takes the gap between its timed neighbours."""
    spans: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
    for owner, hit in zip(owners, aligned):
        if hit is not None:
            spans.setdefault(owner, []).append(hit)
    words = []
    for li, line in enumerate(lines):
        for wi, text in enumerate(line.split()):
            hits = spans.get((li, wi))
            if hits:
                words.append(TimedWord(index=len(words), line_index=li, text=text, start=round(hits[0][0], 3),
                                       end=round(hits[-1][1], 3),
                                       confidence=round(sum(h[2] for h in hits) / len(hits), 3)))
            else:
                words.append(TimedWord(index=len(words), line_index=li, text=text, start=-1.0, end=-1.0))
    _fill_gaps(words)
    return words


def _fill_gaps(words: list[TimedWord]) -> None:
    for i, w in enumerate(words):
        if w.confidence is not None:
            continue
        before = next((p.end for p in reversed(words[:i]) if p.confidence is not None), 0.0)
        after = next((n.start for n in words[i + 1:] if n.confidence is not None), before)
        w.start, w.end = before, max(before, after)


def lyric_lines(lines: list[str], words: list[TimedWord]) -> list[LyricLine]:
    out = []
    for li, text in enumerate(lines):
        own = [w for w in words if w.line_index == li]
        scored = [w.confidence for w in own if w.confidence is not None]
        confidence = round(sum(scored) / len(scored), 3) if scored else 0.0
        out.append(LyricLine(index=li, text=text, start=own[0].start, end=own[-1].end, confidence=confidence,
                             unclear=confidence < UNCLEAR_BELOW))
    return out


def song_confidence(lines: list[LyricLine]) -> float:
    """Mean line confidence, weighted by words so a long line counts more than 'Oh!'."""
    weights = [len(l.text.split()) for l in lines]
    return round(sum(l.confidence * w for l, w in zip(lines, weights)) / max(sum(weights), 1), 3)
