import re

from src.schema.lesson.lesson_schema import Beat, BeatType, GradeBand
from src.schema.song.song_analysis_schema import TimedWord
from src.schema.storyboard.script_schema import (Action, BeatScript, BeatScriptDraft, Drawing, DrawingKind,
                                                 DrawingList, Pos, Screen, Verb)
from src.service.lesson.lesson_rules import band_rules, sung_at

# Each drawing is made once by the image model, so the list is kept short.
MAX_DRAWINGS = 24
# A performer that acts the lesson out is drawn in each state the lesson shows, and in the steps of a change.
MAX_PERFORMER_POSES = 12
MAX_ON_SCREEN = 4
# A row of stages holds up to this many drawings, and stands where left, center and right are.
MAX_ROW = 5
MIDDLE = {Pos.LEFT, Pos.CENTER, Pos.RIGHT}
PALETTE_COLOURS = (4, 6)
ZOOM_SCALE = (0.5, 2.5)
# A beat longer than this needs at least this many actions, so the picture keeps changing.
LONG_BEAT_SECONDS = 4.0
MIN_LONG_BEAT_ACTIONS = 2
# How long a key word or figure stays up once it is sung.
KEYWORD_SECONDS = 2.0
# Times this close to a beat's edge count as inside it.
SLACK = 0.05
# An action this soon after a beat's start is moved to the start, so the screen is never empty for a blink.
SNAP = 0.25

CAMERA = {Verb.PUSH_IN, Verb.PULL_BACK, Verb.PAN}
BY_CODE = {Verb.KEYWORD, Verb.NUMBER, Verb.LIPSYNC}
LASTING = {Verb.MOVE, Verb.ZOOM, Verb.HIGHLIGHT} | CAMERA

_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
_NOT_LETTER = re.compile(r"[^a-z0-9]")


def style_of(band: GradeBand) -> str:
    """The look every drawing of a video is made in."""
    rules = band_rules(band)
    return f"{rules.style_name} with {rules.style}"


def drawing_problems(listed: DrawingList, beats: list[Beat]) -> list[str]:
    """What the schema cannot check in the list of drawings. An empty list means it is usable."""
    found = []
    by_id = {d.id: d for d in listed.drawings}
    if len(by_id) != len(listed.drawings):
        found.append("every drawing needs its own id")
    if len(listed.drawings) > MAX_DRAWINGS:
        found.append(f"{len(listed.drawings)} drawings; at most {MAX_DRAWINGS}, so reuse drawings across beats")
    if not PALETTE_COLOURS[0] <= len(listed.palette) <= PALETTE_COLOURS[1] \
            or not all(_HEX.match(c) for c in listed.palette):
        found.append(f"palette must be {PALETTE_COLOURS[0]} to {PALETTE_COLOURS[1]} hex colours such as #FFC857")
    kinds = [d.kind for d in listed.drawings]
    poses = kinds.count(DrawingKind.PERFORMER)
    if DrawingKind.BACKGROUND not in kinds:
        found.append("list at least one background")
    if poses > MAX_PERFORMER_POSES:
        found.append(f"{poses} performer poses; at most {MAX_PERFORMER_POSES}")
    if any(b.type == BeatType.HERO for b in beats) and not any(d.sings for d in listed.drawings):
        found.append("the song has hero beats, sung to the viewer; list at least one performer drawing that sings")
    if poses and not (listed.performer or "").strip():
        found.append("performer must say who sings the song and what they look like")
    for d in listed.drawings:
        if not _ID.match(d.id):
            found.append(f"id {d.id!r} must be snake_case, starting with a letter")
        if not d.look.strip():
            found.append(f"{d.id} needs a look: what to draw")
        if d.kind in (DrawingKind.BACKGROUND, DrawingKind.ICON) and d.motion:
            found.append(f"{d.id} is a {d.kind} and does not move; give its motion as null")
        if d.sings and d.kind != DrawingKind.PERFORMER:
            found.append(f"{d.id} is a {d.kind}; only a performer drawing sings")
        if d.kind == DrawingKind.PERFORMER and not d.motion:
            found.append(f"{d.id} needs a motion: how the performer moves in this pose")
        if d.of is not None:
            base = by_id.get(d.of)
            if base is None or d.of == d.id or base.kind != d.kind or base.of is not None:
                found.append(f"{d.id}'s of {d.of!r} must be another {d.kind} drawing whose own of is null: the one "
                             "first drawing of that body; give null when nothing looks like it")
    return found


def play(scripts: list[BeatScriptDraft], beats: list[Beat], drawings: dict[str, Drawing], before: Screen
         ) -> tuple[list[str], list[Screen]]:
    """The scripts for a run of beats played in order, starting from the screen `before`: what is wrong with
    them, and the screen as each beat ends."""
    if [s.beat for s in scripts] != [b.id for b in beats]:
        return [f"give exactly one script for each of beats {beats[0].id} to {beats[-1].id}, in order"], []
    found, screens = [], []
    for script, beat in zip(scripts, beats):
        problems, before = _play_beat(script, beat, drawings, before)
        found += problems
        screens.append(before)
    return found, screens


def _play_beat(script: BeatScriptDraft, beat: Beat, drawings: dict[str, Drawing], before: Screen
               ) -> tuple[list[str], Screen]:
    """One beat: its drawings stay from the beat before unless the background changes, which clears the
    screen. Something must really move in it, and a hero beat keeps its singer on screen throughout. Actions
    just after the beat's start are moved to the start."""
    found = []
    for a in script.actions:
        if beat.start - SLACK <= a.at <= beat.start + SNAP:
            a.at = beat.start
    back = drawings.get(script.background)
    if back is None or back.kind != DrawingKind.BACKGROUND:
        found.append(f"{beat.id}: background {script.background!r} is not a listed background")
    on = dict(before.on) if script.background == before.background else {}
    acts = sorted(script.actions, key=lambda a: a.at)
    opening = [a for a in acts if a.at <= beat.start + SLACK]
    for a in opening:
        found += _apply(a, beat, drawings, on)
    singer = next((d for d in on if drawings[d].kind == DrawingKind.PERFORMER), None)
    alive = _alive(on, drawings)
    for a in acts[len(opening):]:
        if beat.type == BeatType.HERO and singer and a.drawing == singer and a.verb in (Verb.HIDE, Verb.SWAP):
            found.append(f"{beat.id}: {singer} sings the whole hero beat; do not hide or swap it")
        found += _apply(a, beat, drawings, on)
        alive = alive or _alive(on, drawings)
    if not alive and not any(a.verb == Verb.SWAP for a in acts):
        found.append(f"{beat.id}: nothing really moves; keep a drawing that has a motion on screen, or swap one "
                     "drawing for another")
    seconds = beat.end - beat.start
    if seconds > LONG_BEAT_SECONDS and len(acts) < MIN_LONG_BEAT_ACTIONS:
        found.append(f"{beat.id} lasts {seconds:.1f} s with {len(acts)} actions; give it at least "
                     f"{MIN_LONG_BEAT_ACTIONS} so the picture keeps changing")
    if beat.type != BeatType.HERO:
        singer = None
    elif singer is None:
        found.append(f"{beat.id} is a hero beat: a performer drawing must be on screen from its start, "
                     f"{beat.start:g}")
    elif not drawings[singer].sings:
        found.append(f"{beat.id} is a hero beat but {singer} cannot sing; have a performer drawing that sings on "
                     "screen from its start")
    return found, Screen(background=script.background, on=on, singer=singer)


def _alive(on: dict[str, Pos], drawings: dict[str, Drawing]) -> bool:
    return any(drawings[d].motion for d in on)


def _apply(a: Action, beat: Beat, drawings: dict[str, Drawing], on: dict[str, Pos]) -> list[str]:
    """One action applied to what is on screen; returns what is wrong with it. Drawings in the row stand
    apart from the rest: several performer drawings may be in it, as the stages of one life."""
    what = f"{beat.id} {a.verb} at {a.at:g}"
    if a.verb in BY_CODE:
        return [f"{what}: our code adds {a.verb}; leave it out"]
    found = []
    if not beat.start - SLACK <= a.at <= beat.end + SLACK:
        found.append(f"{what}: at must be inside the beat, {beat.start:g} to {beat.end:g}")
    if a.verb in LASTING and (a.until is None or not a.at < a.until <= beat.end + SLACK):
        found.append(f"{what}: until must be after at and no later than {beat.end:g}")
    if a.verb in CAMERA:
        if a.verb == Verb.PAN and a.pos not in (Pos.LEFT, Pos.RIGHT):
            found.append(f"{what}: pos must be left or right")
        return found
    thing = drawings.get(a.drawing or "")
    if thing is None or thing.kind == DrawingKind.BACKGROUND:
        return found + [f"{what}: drawing {a.drawing!r} is not a listed performer, object or icon"]
    row = [d for d, p in on.items() if p == Pos.ROW and d != a.drawing]
    rest = [d for d, p in on.items() if p != Pos.ROW and d != a.drawing]
    taken = next((d for d in rest if on[d] == a.pos), None)
    if a.pos == Pos.ROW:                                        # a row needs the middle of the screen to itself
        taken = next((d for d in rest if on[d] in MIDDLE), None)
    elif a.pos in MIDDLE and row:
        taken = row[0]
    other = next((d for d in rest if drawings[d].kind == DrawingKind.PERFORMER), None)
    twice = f"{what}: {other} is on screen; only one performer drawing at a time, so swap that one instead"
    spot = f"{what}: {a.pos} is taken by {taken}; hide or move it first" if taken else f"{what}: pos is needed"
    if a.verb == Verb.SHOW:
        if a.drawing in on:
            found.append(f"{what}: {a.drawing} is already on screen; move it instead")
        elif thing.kind == DrawingKind.PERFORMER and other and a.pos != Pos.ROW:
            found.append(twice)
        elif a.pos is None or taken:
            found.append(spot)
        elif a.pos == Pos.ROW and len(row) == MAX_ROW:
            found.append(f"{what}: the row is full; at most {MAX_ROW} stages in a row")
        else:
            on[a.drawing] = a.pos
            if len(on) - len(row) > MAX_ON_SCREEN:
                found.append(f"{what}: {len(on)} drawings are on screen; at most {MAX_ON_SCREEN}, so hide one first")
    elif a.drawing not in on:
        found.append(f"{what}: {a.drawing} is not on screen; show it first")
    elif a.verb == Verb.HIDE:
        del on[a.drawing]
    elif a.verb == Verb.SWAP:
        new = drawings.get(a.to or "")
        if new is None or new.kind == DrawingKind.BACKGROUND or a.to in on:
            found.append(f"{what}: to {a.to!r} must be a listed performer, object or icon that is not on screen")
        elif new.kind == DrawingKind.PERFORMER and other:
            found.append(twice)
        else:
            on[a.to] = on.pop(a.drawing)
    elif a.verb == Verb.MOVE:
        if a.pos == Pos.ROW:
            found.append(f"{what}: a drawing joins the row with show, not move")
        elif a.pos is None or taken:
            found.append(spot)
        else:
            on[a.drawing] = a.pos
    elif a.verb == Verb.ZOOM and (a.scale is None or not ZOOM_SCALE[0] <= a.scale <= ZOOM_SCALE[1]):
        found.append(f"{what}: scale must be between {ZOOM_SCALE[0]} and {ZOOM_SCALE[1]}")
    return found


def finish(scripts: list[BeatScriptDraft], beats: list[Beat], screens: list[Screen], words: list[TimedWord],
           duration: float, terms: list[str]) -> list[BeatScript]:
    """The checked scripts with their beats' timing and code's actions added: the key word and the figure
    as they are sung, each starting inside its beat and held for KEYWORD_SECONDS even past the beat's end,
    and a hero beat's lip-sync. A key word sung in pieces (Met-a-MOR-pho-SIS) is written as its key term."""
    spelling = {_NOT_LETTER.sub("", t.lower()): t for t in terms}
    out = []
    for script, beat, screen in zip(scripts, beats, screens):
        extra = []
        if beat.keyword:
            text = spelling.get(_NOT_LETTER.sub("", beat.keyword.lower()), beat.keyword)
            extra.append(_text(Verb.KEYWORD, text, beat.keyword_at, beat, duration))
        if beat.number_text and beat.number_text.lower() != (beat.keyword or "").lower():
            sung = [w for w in words if w.line_index in beat.lines]
            extra.append(_text(Verb.NUMBER, beat.number_text, sung_at(beat.number_text, sung), beat, duration))
        if screen.singer:
            extra.append(Action(verb=Verb.LIPSYNC, drawing=screen.singer, at=beat.start, until=beat.end))
        out.append(BeatScript(beat=beat.id, background=script.background,
                              actions=sorted(script.actions + extra, key=lambda a: a.at),
                              start=beat.start, end=beat.end, type=beat.type))
    return out


def _text(verb: Verb, text: str, at: float | None, beat: Beat, duration: float) -> Action:
    at = beat.start if at is None else min(max(at, beat.start), beat.end)
    return Action(verb=verb, text=text, at=at, until=round(min(at + KEYWORD_SECONDS, duration), 3))
