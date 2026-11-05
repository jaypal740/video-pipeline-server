import re
import tomllib
from difflib import SequenceMatcher
from pathlib import Path

from src.schema.lesson.lesson_schema import BandRules, FormatRules, GradeBand, LessonDraft, LineTag, VideoType
from src.schema.song.song_analysis_schema import LyricLine

TEMPLATES = Path(__file__).resolve().parents[2] / "templates"
# A line this close to an earlier one of its section, by words, is that line sung again.
REPEAT_SIMILAR = 0.8
_AD_LIB = re.compile(r"\([^)]*\)")
_WORD = re.compile(r"[a-z0-9']+")
# A sequence is one topic the viewer watches unfold; shorter than this it is just another picture,
# longer than this it holds more than one topic.
MIN_SEQUENCE_SECONDS = 8.0
MAX_SEQUENCE_SECONDS = 40.0
MAX_SEQUENCES = 8


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


def allowed_types(band: GradeBand, requested: VideoType | None) -> list[VideoType]:
    """The types the model may choose from: the band's list, or the one type a product asked for."""
    rules = band_rules(band)
    if requested is None:
        return rules.allowed_types
    if requested not in rules.allowed_types + rules.override_types:
        raise ValueError(f"Video type {requested} is not allowed for grades {band}")
    return [requested]


def face_limit(band: GradeBand, video_type: VideoType, line_count: int) -> int:
    share = min(band_rules(band).face_to_camera_share, format_rules(video_type).face_to_camera_share)
    return int(share * line_count)


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


def sequence_problems(draft: LessonDraft, lines: list[LyricLine], featured: list[str]) -> list[str]:
    """Sequences cover every line once in order, are long enough to show a topic unfolding, keep a chorus
    sung again in its first place, and name the terms their lines sing."""
    seqs = draft.sequences
    if not seqs:
        return ["sequences must group every line into topic sequences"]
    found = []
    if len(seqs) > MAX_SEQUENCES:
        found.append(f"{len(seqs)} sequences; at most {MAX_SEQUENCES} are allowed")
    indexes = [l.index for l in lines]
    covered = [i for q in seqs for i in range(q.first_line, q.last_line + 1)]
    if covered != indexes:
        found.append(f"sequences must cover lines {indexes[0]}-{indexes[-1]} once each, in order, without gaps")
        return found
    by_index = {l.index: l for l in lines}
    home = {i: q for q in seqs for i in range(q.first_line, q.last_line + 1)}
    for q in seqs:
        text = " ".join(by_index[i].text.lower() for i in range(q.first_line, q.last_line + 1))
        seconds = by_index[q.last_line].end - by_index[q.first_line].start
        if len(seqs) > 1 and seconds < MIN_SEQUENCE_SECONDS:
            found.append(f"sequence {q.first_line}-{q.last_line} lasts {seconds:.1f} s; join it to a neighbour "
                         f"so it lasts at least {MIN_SEQUENCE_SECONDS:.0f} s")
        if seconds > MAX_SEQUENCE_SECONDS:
            found.append(f"sequence {q.first_line}-{q.last_line} lasts {seconds:.1f} s; split it by topic so "
                         f"each part lasts at most {MAX_SEQUENCE_SECONDS:.0f} s")
        missing = [t for t in q.terms if t.lower() not in text]
        if missing:
            found.append(f"sequence {q.first_line}-{q.last_line} lists terms its lines don't sing: {missing}")
    for index, first in repeats(lines).items():
        if home[index].place != home[first].place:
            found.append(f"line {index} repeats line {first}; put its sequence in the same place, "
                         f"{home[first].place!r}")
    listed = {t.lower() for q in seqs for t in q.terms}
    sung = " ".join(l.text.lower() for l in lines)
    unlisted = [t for t in featured if t.lower() in sung and not any(t.lower() in l for l in listed)]
    if unlisted:
        found.append(f"featured terms {unlisted} are sung but no sequence lists them")
    return found


def problems(draft: LessonDraft, lines: list[LyricLine], band: GradeBand, types: list[VideoType],
             featured: list[str] | None = None) -> list[str]:
    """What the schema cannot check. An empty list means the draft is usable."""
    found = []
    if [l.index for l in draft.lines] != [l.index for l in lines]:
        found.append(f"lines must be exactly indexes {[l.index for l in lines]} in order, one entry each")
    if draft.video_type not in types:
        found.append(f"video_type must be one of {[str(t) for t in types]}")
    if not draft.key_terms:
        found.append("key_terms must list the terms the song teaches")
    by_index = {l.index: l for l in lines}
    by_entry = {e.index: e for e in draft.lines}
    for index, first in repeats(lines).items():
        entry, original = by_entry.get(index), by_entry.get(first)
        if entry and original and (entry.teaches, entry.show, entry.tag) != (original.teaches, original.show, original.tag):
            found.append(f"line {index} repeats line {first}; give it the same teaches, show and tag")
    faces = 0
    for entry in draft.lines:
        line = by_index.get(entry.index)
        if line is None:
            continue
        if entry.tag == LineTag.UNCLEAR and not line.unclear:
            found.append(f"line {entry.index}: only lines marked [unclear] may be tagged unclear")
        if entry.tag == LineTag.NUMBER and not (entry.number_text and entry.number_text.lower() in line.text.lower()):
            found.append(f"line {entry.index}: a number line needs number_text copied exactly from the line")
        faces += entry.tag == LineTag.FACE_TO_CAMERA
    if draft.video_type in types:
        limit = face_limit(band, draft.video_type, len(lines))
        if faces > limit:
            found.append(f"{faces} lines are face_to_camera; at most {limit} are allowed for this band and type")
    return found + sequence_problems(draft, lines, featured or [])
