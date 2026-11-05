import math

from src.schema.lesson.lesson_schema import LessonLine, LessonSequence, LineTag
from src.schema.song.song_analysis_schema import LyricLine
from src.schema.storyboard.storyboard_schema import ShotKind, Slot
from src.service.lesson.lesson_rules import repeats

# Lip-sync models take at most 10 s of video, so a sung shot is never longer.
SUNG_MAX = 10.0
# A cut moves to a beat this close to it.
SNAP = 0.25
MIN_SLOT = 0.5

_KIND = {LineTag.FACE_TO_CAMERA: ShotKind.SUNG, LineTag.NUMBER: ShotKind.NUMBER,
         LineTag.ILLUSTRATE: ShotKind.ILLUSTRATE, LineTag.UNCLEAR: ShotKind.UNCLEAR}


class _Span:
    """A stretch of the song that becomes one or more shots. repeat_line: the earlier line this one repeats
    (a chorus sung again); original: a line that a later line repeats; seq: its lesson sequence."""

    def __init__(self, start: float, end: float, lines: list[int], kind: ShotKind, number_text: str | None,
                 repeat_line: int | None, original: bool = False, seq: int = 0) -> None:
        self.start, self.end, self.lines, self.kind = start, end, lines, kind
        self.number_text, self.repeat_line, self.original, self.seq = number_text, repeat_line, original, seq

    def seconds(self) -> float:
        return self.end - self.start


def _snap(t: float, beats: list[float], earlier_only: bool) -> float:
    near = [b for b in beats if (t - SNAP <= b <= t if earlier_only else abs(b - t) <= SNAP)]
    return min(near, key=lambda b: abs(b - t)) if near else t


def _spans(lines: list[LyricLine], lesson: dict[int, LessonLine], beats: list[float], duration: float,
           low: float, seq_of: dict[int, int]) -> list[_Span]:
    """One span per line, from its start to the next line's start. A gap without lyrics long enough for a
    shot of its own (intro, break, outro) becomes a music span; a shorter one joins the line before it."""
    again = repeats(lines)
    originals = set(again.values())
    spans = []
    for i, line in enumerate(lines):
        start = _snap(line.start, beats, earlier_only=True)
        if i == 0 and start < low:                      # an intro too short for a shot joins the first line
            start = 0.0
        entry = lesson[line.index]
        spans.append(_Span(start, line.end, [line.index], _KIND[entry.tag], entry.number_text,
                           again.get(line.index), line.index in originals, seq_of.get(line.index, 0)))
    spans = _absorb_tiny(spans)
    joined = []
    for a, b in zip(spans, spans[1:] + [None]):
        joined.append(a)
        if b is None:
            continue
        if b.start - a.end >= low:
            joined.append(_Span(a.end, b.start, [], ShotKind.MUSIC, None, None, seq=b.seq))
        else:
            a.end = b.start
    spans = joined
    if spans[0].start > 0:
        spans.insert(0, _Span(0.0, spans[0].start, [], ShotKind.MUSIC, None, None, seq=spans[0].seq))
    if duration - spans[-1].end >= low:
        spans.append(_Span(spans[-1].end, duration, [], ShotKind.MUSIC, None, None, seq=spans[-1].seq))
    else:
        spans[-1].end = max(spans[-1].end, duration)
    return spans


def _absorb_tiny(spans: list[_Span]) -> list[_Span]:
    """A line too short to be a shot (intake can time one at almost zero length) joins its neighbour."""
    out: list[_Span] = []
    carry: _Span | None = None
    for span in spans:
        if carry is not None:
            span.start, span.lines = carry.start, carry.lines + span.lines
            carry = None
        if span.seconds() >= MIN_SLOT:
            out.append(span)
        elif out:
            out[-1].end, out[-1].lines = max(out[-1].end, span.end), out[-1].lines + span.lines
        else:
            carry = span
    return out


def _same_original(a: _Span, b: _Span, out: list[_Span]) -> bool:
    """Two repeated lines whose originals share a shot: the repeat is cut the way the original was."""
    home = {line: i for i, span in enumerate(out) for line in span.lines}
    return home.get(a.repeat_line) is not None and home.get(a.repeat_line) == home.get(b.repeat_line)


def _merge_short(spans: list[_Span], low: float, high: float) -> list[_Span]:
    """A picture line shorter than a shot shares a shot with the next picture line of its sequence when both
    fit in one. Sung, number and music lines keep their own shots. A chorus line merges only with another
    chorus line, and its repeats merge exactly as the originals did, so a repeat copies one whole shot."""
    def picture(s: _Span) -> bool:
        return s.kind in (ShotKind.ILLUSTRATE, ShotKind.UNCLEAR)

    out: list[_Span] = []
    for span in spans:
        prev = out[-1] if out else None
        if prev is None or not (picture(prev) and picture(span)) or prev.seq != span.seq:
            out.append(span)
        elif prev.repeat_line is not None or span.repeat_line is not None:
            if prev.repeat_line is not None and span.repeat_line is not None and _same_original(prev, span, out):
                prev.end, prev.lines = span.end, prev.lines + span.lines
            else:
                out.append(span)
        elif prev.original == span.original and prev.seconds() < low and span.end - prev.start <= high:
            prev.end, prev.lines = span.end, prev.lines + span.lines
        else:
            out.append(span)
    return out


def _pieces(span: _Span, beats: list[float], low: float, high: float) -> list[tuple[float, float]]:
    """Cut a span into shots within the band's length; never into shots shorter than the band minimum, so a
    span just over the maximum stays one shot. Sung and number spans stay whole up to SUNG_MAX."""
    if span.kind in (ShotKind.SUNG, ShotKind.NUMBER):
        count = math.ceil(span.seconds() / SUNG_MAX - 1e-9)
    else:
        count = min(math.ceil(span.seconds() / high - 1e-9), math.floor(span.seconds() / low + 1e-9))
    count = max(1, count)
    step = span.seconds() / count
    sung = span.kind in (ShotKind.SUNG, ShotKind.NUMBER)
    cuts = [span.start]
    for k in range(1, count):
        plain = span.start + k * step
        cut = _snap(plain, beats, earlier_only=False)
        if cut - cuts[-1] < MIN_SLOT or (sung and max(cut - cuts[-1], plain + step - cut) > SUNG_MAX):
            cut = plain
        cuts.append(cut)
    cuts.append(span.end)
    return list(zip(cuts, cuts[1:]))


def _like(span: _Span, originals: list[Slot]) -> list[tuple[float, float]] | None:
    """A repeat cut into as many shots as its original, at the same proportions, so each copied clip fills
    its own shot; None when that would break the sung or minimum length."""
    first, last = originals[0].start, originals[-1].end
    if last <= first:
        return None
    cuts = [span.start + (o.start - first) / (last - first) * span.seconds() for o in originals] + [span.end]
    pieces = list(zip(cuts, cuts[1:]))
    longest = SUNG_MAX if span.kind in (ShotKind.SUNG, ShotKind.NUMBER) else math.inf
    return None if any(not MIN_SLOT <= b - a <= longest for a, b in pieces) else pieces


def make_slots(lines: list[LyricLine], lesson_lines: list[LessonLine], beats: list[float], duration: float,
               shot_seconds: list[float], sequences: list[LessonSequence] | None = None) -> list[Slot]:
    """Cut the song into shots: each line split to the band's shot length on the beat, sung and number
    lines kept whole up to SUNG_MAX, no shot across two sequences, and a repeated chorus line cut like
    the shots it copies and pointed at them."""
    low, high = shot_seconds
    lesson = {l.index: l for l in lesson_lines}
    seq_of = {i: k for k, q in enumerate(sequences or []) for i in range(q.first_line, q.last_line + 1)}
    spans = _merge_short(_spans(lines, lesson, beats, duration, low, seq_of), low, high)
    slots, by_line, by_id = [], {}, {}
    for span in spans:
        pieces = _pieces(span, beats, low, high)
        if span.repeat_line is not None:
            pieces = _like(span, [by_id[i] for i in by_line[span.repeat_line]]) or pieces
        for k, (start, end) in enumerate(pieces):
            slot = Slot(id=f"s{len(slots) + 1:02d}", start=round(start, 3), end=round(end, 3), lines=span.lines,
                        kind=span.kind, number_text=span.number_text, sequence=span.seq)
            if span.repeat_line is not None:
                originals = by_line[span.repeat_line]
                slot.repeat_of = originals[min(k, len(originals) - 1)]
            for index in span.lines:
                by_line.setdefault(index, []).append(slot.id)
            by_id[slot.id] = slot
            slots.append(slot)
    return slots
