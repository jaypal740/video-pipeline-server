import re
import tomllib
from difflib import SequenceMatcher
from pathlib import Path

from src.schema.lesson.lesson_schema import (BandRules, Beat, BeatDraft, BeatType, FormatRules, GradeBand, LessonDraft,
                                             VideoType)
from src.schema.song.song_analysis_schema import LyricLine, TimedWord

TEMPLATES = Path(__file__).resolve().parents[2] / "templates"
# A line this close to an earlier one of its section, by words, is that line sung again.
REPEAT_SIMILAR = 0.8
_AD_LIB = re.compile(r"\([^)]*\)")
_WORD = re.compile(r"[a-z0-9']+")
_SENTENCE_END = re.compile(r"[.!?]\s+\S")
# A beat teaches one idea: with more lines than this, or longer than this, it holds more than one.
MAX_BEAT_LINES = 4
MAX_BEAT_SECONDS = 12.0
# Lip-sync models take at most 10 s of video, so a hero beat is never longer.
HERO_MAX_SECONDS = 10.0
MAX_CONCEPT_WORDS = 20
MAX_KEYWORD_WORDS = 3
# A stretch without lyrics at least this long (intro, break, outro) is a music beat of its own; a shorter
# one joins the beat beside it.
MIN_MUSIC_SECONDS = 3.0


def band_rules(band: GradeBand) -> BandRules:
    return BandRules(**tomllib.loads((TEMPLATES / "band" / f"{band}.toml").read_text()))


def format_rules(video_type: VideoType) -> FormatRules:
    return FormatRules(**tomllib.loads((TEMPLATES / "format" / f"{video_type}.toml").read_text()))


def band_for(grade: int) -> GradeBand:
    """The band a grade falls in; 0 is kindergarten. A jam spanning bands uses its lowest grade, so the
    video is never too old for its youngest viewer."""
    for band in GradeBand:
        if grade in band_rules(band).grades:
            return band
    raise ValueError(f"Grade {grade} is outside K-8")


def hero_limit(band: GradeBand, line_count: int) -> int:
    """How many lines the performer may sing to the viewer: the band's share of the song."""
    return int(band_rules(band).face_to_camera_share * line_count)


def _words(text: str) -> list[str]:
    return _WORD.findall(_AD_LIB.sub(" ", text.lower()))


def repeats(lines: list[LyricLine]) -> dict[int, int]:
    """Lines that repeat an earlier line of the same section (a chorus sung again, ad-libs and punctuation
    aside), mapped to that earlier line's index. Untagged and unclear lines never count as repeats."""
    first: list[LyricLine] = []
    found = {}
    for line in lines:
        if line.section is None or line.unclear:
            continue
        words = _words(line.text)
        score, match = max(((SequenceMatcher(None, words, _words(f.text)).ratio(), f.index)
                            for f in first if f.section == line.section), key=lambda m: m[0], default=(0.0, None))
        if score >= REPEAT_SIMILAR:
            found[line.index] = match
        else:
            first.append(line)
    return found


def _plain(text: str) -> str:
    return " " + " ".join(_WORD.findall(text.lower())) + " "


def beat_times(beats: list[BeatDraft], lines: list[LyricLine], duration: float) -> list[tuple[float, float]]:
    """Each beat's (start, end) from its lines' timing. A beat runs to the next beat's start, and the first
    and last reach the song's ends, unless the stretch without lyrics there is long enough for a music beat."""
    by_index = {l.index: l for l in lines}
    starts = [by_index[b.first_line].start for b in beats]
    ends = [by_index[b.last_line].end for b in beats]
    times = []
    for i, (start, end) in enumerate(zip(starts, ends)):
        if i == 0 and start < MIN_MUSIC_SECONDS:
            start = 0.0
        until = starts[i + 1] if i + 1 < len(beats) else duration
        times.append((round(start, 3), round(until if until - end < MIN_MUSIC_SECONDS else end, 3)))
    return times


def sung_at(keyword: str, words: list[TimedWord]) -> float | None:
    """When the key word starts being sung: the start of the first timed word that begins it."""
    want = _WORD.findall(keyword.lower())
    tokens = [(token, w.start) for w in words for token in _WORD.findall(w.text.lower())]
    for i in range(len(tokens) - len(want) + 1):
        if [token for token, _ in tokens[i:i + len(want)]] == want:
            return tokens[i][1]
    return None


def timed_beats(beats: list[BeatDraft], lines: list[LyricLine], words: list[TimedWord], duration: float) -> list[Beat]:
    """The beats in time, with music beats filling the stretches without lyrics so the song is covered
    without gaps, and the moment each key word is sung, kept inside its beat when a word's timing strays."""
    unclear = {l.index for l in lines if l.unclear}
    out: list[Beat] = []

    def add(**fields) -> None:
        out.append(Beat(id=f"b{len(out) + 1:02d}", **fields))

    at = 0.0
    for beat, (start, end) in zip(beats, beat_times(beats, lines, duration)):
        if start > at:
            add(start=at, end=start, type=BeatType.MUSIC)
        own = list(range(beat.first_line, beat.last_line + 1))
        keyword = beat.keyword or None
        sung = [w for w in words if w.line_index in own and w.line_index not in unclear]
        at = sung_at(keyword, sung) if keyword else None
        add(start=start, end=end, type=beat.type, lines=own, concept=beat.concept, keyword=keyword,
            keyword_at=None if at is None else min(max(at, start), end), number_text=beat.number_text or None,
            avoid=beat.avoid, hint=beat.hint)
        at = end
    if duration > at:
        add(start=at, end=round(duration, 3), type=BeatType.MUSIC)
    return out


def problems(draft: LessonDraft, lines: list[LyricLine], band: GradeBand, duration: float) -> list[str]:
    """What the schema cannot check. An empty list means the draft is usable."""
    found = []
    if not draft.key_terms:
        found.append("key_terms must list the terms the song teaches")
    indexes = [l.index for l in lines]
    if [i for b in draft.beats for i in range(b.first_line, b.last_line + 1)] != indexes:
        return found + [f"beats must cover lines {indexes[0]}-{indexes[-1]} once each, in order, without gaps"]
    by_index = {l.index: l for l in lines}
    again = repeats(lines)
    hero_lines = 0
    for beat, (start, end) in zip(draft.beats, beat_times(draft.beats, lines, duration)):
        name = f"beat {beat.first_line}-{beat.last_line}"
        own = [by_index[i] for i in range(beat.first_line, beat.last_line + 1)]
        clear = " ".join(l.text for l in own if not l.unclear)
        seconds = end - start
        if beat.type == BeatType.MUSIC:
            found.append(f"{name} is a music beat; code adds those, so give it another type")
        if len(own) > MAX_BEAT_LINES:
            found.append(f"{name} has {len(own)} lines; a beat teaches one idea in at most {MAX_BEAT_LINES} lines, "
                         "so split it")
        elif len(own) > 1 and seconds > MAX_BEAT_SECONDS:
            found.append(f"{name} lasts {seconds:.1f} s; split it so each part lasts at most {MAX_BEAT_SECONDS:.0f} s")
        if not 0 < len(beat.concept.split()) <= MAX_CONCEPT_WORDS or _SENTENCE_END.search(beat.concept.strip()):
            found.append(f"{name}'s concept must be one sentence of at most {MAX_CONCEPT_WORDS} words; split the "
                         "beat if it teaches two ideas")
        if not beat.hint.strip():
            found.append(f"{name} needs a hint: the few things to draw and what happens to them")
        if beat.keyword:
            if not 0 < len(beat.keyword.split()) <= MAX_KEYWORD_WORDS:      # Met-a-MOR-pho-SIS is one word
                found.append(f"{name}'s keyword {beat.keyword!r} must be 1 to {MAX_KEYWORD_WORDS} words")
            elif _plain(beat.keyword) not in _plain(clear):
                found.append(f"{name}'s keyword {beat.keyword!r} is not sung in its lines; copy a word exactly as "
                             "it is sung there, not from an [unclear] line, or give null")
        if beat.number_text and beat.number_text.lower() not in clear.lower():
            found.append(f"{name}'s number_text {beat.number_text!r} must be copied exactly from its lines, not "
                         "from an [unclear] line")
        if beat.type == BeatType.HERO:
            hero_lines += len(own)
            if seconds > HERO_MAX_SECONDS:
                found.append(f"{name} is a hero beat lasting {seconds:.1f} s; lip-sync takes at most "
                             f"{HERO_MAX_SECONDS:.0f} s, so split it or give it another type")
            if any(l.unclear for l in own):
                found.append(f"{name} is a hero beat with an [unclear] line; the performer sings only clear lines")
        elif beat.type != BeatType.RECAP:
            repeated = next((l.index for l in own if l.index in again), None)
            if repeated is not None:
                found.append(f"line {repeated} repeats line {again[repeated]}; make {name} a recap or hero beat")
    limit = hero_limit(band, len(lines))
    if hero_lines > limit:
        found.append(f"{hero_lines} lines are in hero beats; at most {limit} are allowed for grades {band}")
    return found
