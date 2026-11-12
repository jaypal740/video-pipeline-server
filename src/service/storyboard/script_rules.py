import re

from src.schema.lesson.lesson_schema import Beat, BeatType, GradeBand
from src.schema.song.song_analysis_schema import TimedWord
from src.schema.storyboard.script_schema import (Action, BeatScript, BeatScriptDraft, Drawing, DrawingKind,
                                                 DrawingList, Pos, Screen, ScriptRepair, Verb)
from src.service.lesson.lesson_rules import band_rules, sung_at
from src.service.render.timeline import covers, scenes_of

MAX_ON_SCREEN = 4
# A look is drawn from its words, and an image model draws whatever is named, so a look names only what is seen.
ABSENT = ("no", "not", "without")
# A row or a ring of stages holds up to this many drawings. The row stands where left, center and right
# are; the ring goes round the screen and leaves only its centre free.
MAX_ROW = 5
KEPT_CLEAR = {Pos.ROW: {Pos.LEFT, Pos.CENTER, Pos.RIGHT}, Pos.RING: {Pos.LEFT, Pos.RIGHT, Pos.TOP, Pos.BOTTOM}}
PALETTE_COLOURS = (4, 6)
# A stage's name, written under its drawing in a row or ring, is this many words at most.
MAX_NAME_WORDS = 2
# The smallest a zoom makes a drawing, and the biggest each kind can be and still fit the screen.
ZOOM_LEAST = 0.5
ZOOM_MOST = {DrawingKind.PERFORMER: 1.5, DrawingKind.OBJECT: 2.0, DrawingKind.ICON: 2.5}
# The longest a beat's picture may hold still: within this many seconds a drawing is shown, hidden, swapped,
# moved or zoomed. A highlight or a camera move is not enough.
MAX_STILL = 4.0
# How long a key word or figure stays up once it is sung.
KEYWORD_SECONDS = 2.0
# Times this close to a beat's edge count as inside it.
SLACK = 0.05
# An action this soon after a beat's start is moved to the start, so the screen is never empty for a blink.
SNAP = 0.25

CAMERA = {Verb.PUSH_IN, Verb.PULL_BACK, Verb.PAN}
BY_CODE = {Verb.KEYWORD, Verb.NUMBER, Verb.LIPSYNC}
LASTING = {Verb.MOVE, Verb.ZOOM, Verb.HIGHLIGHT} | CAMERA
NEW = {Verb.SHOW, Verb.HIDE, Verb.SWAP, Verb.MOVE, Verb.ZOOM}

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
    if not PALETTE_COLOURS[0] <= len(listed.palette) <= PALETTE_COLOURS[1] \
            or not all(_HEX.match(c) for c in listed.palette):
        found.append(f"palette must be {PALETTE_COLOURS[0]} to {PALETTE_COLOURS[1]} hex colours such as #FFC857")
    kinds = [d.kind for d in listed.drawings]
    poses = kinds.count(DrawingKind.PERFORMER)
    if DrawingKind.BACKGROUND not in kinds:
        found.append("list at least one background")
    if any(b.type == BeatType.HERO for b in beats) and not any(d.sings for d in listed.drawings):
        found.append("the song has hero beats, sung to the viewer; list at least one performer drawing that sings")
    if poses and not (listed.performer or "").strip():
        found.append("performer must say who sings the song and what they look like")
    for d in listed.drawings:
        if not _ID.match(d.id):
            found.append(f"id {d.id!r} must be snake_case, starting with a letter")
        if not d.look.strip():
            found.append(f"{d.id} needs a look: what to draw")
        if absent := [w for w in ABSENT if w in re.findall(r"[a-z]+", d.look.lower())]:
            found.append(f"{d.id}'s look says what is not there ({absent[0]!r}); whatever a look names gets drawn, so "
                         "say only what is seen")
        if d.name and (d.kind == DrawingKind.BACKGROUND or len(d.name.split()) > MAX_NAME_WORDS):
            found.append(f"{d.id}'s name {d.name!r} must be a stage's name of at most {MAX_NAME_WORDS} words, on a "
                         "drawing that is not a background; give null otherwise")
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


def mend(listed: DrawingList, scripts: list[BeatScriptDraft], repair: ScriptRepair
         ) -> tuple[DrawingList, list[BeatScriptDraft]]:
    """The drawings and the script with a repair applied: its new drawings added, its beats in place of the old."""
    new = {s.beat: s for s in repair.beats}
    return (listed.model_copy(update={"drawings": listed.drawings + repair.drawings}),
            [new.get(s.beat, s) for s in scripts])


def repair_problems(repair: ScriptRepair, listed: DrawingList, scripts: list[BeatScriptDraft], beats: list[Beat]
                    ) -> list[str]:
    """What is wrong with a repair: the whole script must still play with it, from its first beat."""
    stray = sorted({s.beat for s in repair.beats} - {s.beat for s in scripts})
    if stray or not repair.beats:
        return [f"give the beats that change, by their ids in the script; {', '.join(stray) or 'none'} given"]
    mended, fixed = mend(listed, scripts, repair)
    return drawing_problems(mended, beats) + play(fixed, beats, {d.id: d for d in mended.drawings}, Screen())[0]


def cover_problems(listed: DrawingList, scripts: list[BeatScript], wide: float, since: float = 0.0) -> list[str]:
    """Where, from `since` on, one drawing would hide much of another in the film made from these scripts, as
    code lays them out on a frame `wide` times as wide as it is tall."""
    return [f"{next(s.beat for s in scripts if s.start <= at < s.end)}: from {at:g} {front} covers {behind}; nothing "
            "may hide another drawing, so put them at places that stand clear (top and bottom overlap center, and a "
            "drawing zoomed big overlaps its neighbours), make one smaller, or hide one first"
            for at, front, behind in covers(scenes_of(listed, scripts, wide), wide) if at >= since]


def _play_beat(script: BeatScriptDraft, beat: Beat, drawings: dict[str, Drawing], before: Screen
               ) -> tuple[list[str], Screen]:
    """One beat: its drawings stay from the beat before unless the background changes, which clears the
    screen. Something must really move in it, its picture never holds still for long, and a hero beat has
    the performer on screen from its start: whichever of its drawings can sing is the one that sings. Actions
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
    if beat.type == BeatType.HERO and not any(drawings[d].kind == DrawingKind.PERFORMER for d in on):
        found.append(f"{beat.id} is a hero beat: a performer drawing must be on screen from its start, "
                     f"{beat.start:g}")
    alive, voices = _alive(on, drawings), [(beat.start, _singer(on, drawings))]
    for a in acts[len(opening):]:
        found += _apply(a, beat, drawings, on)
        alive = alive or _alive(on, drawings)
        if _singer(on, drawings) != voices[-1][1]:
            voices.append((a.at, _singer(on, drawings)))
    if not alive and not any(a.verb == Verb.SWAP for a in acts):
        found.append(f"{beat.id}: nothing really moves; keep a drawing that has a motion on screen, or swap one "
                     "drawing for another")
    last = beat.start
    for at, until in [(a.at, a.until or a.at) for a in acts if a.verb in NEW] + [(beat.end, beat.end)]:
        if at - last > MAX_STILL + SLACK:
            found.append(f"{beat.id}: nothing new happens from {last:g} to {at:g}; the picture holds still for at "
                         f"most {MAX_STILL:g} s, so show, swap, move or zoom a drawing there: the next step of the "
                         "beat's idea")
            break
        last = max(last, until)
    sung = [(who, at, until) for (at, who), (until, _) in zip(voices, voices[1:] + [(beat.end, None)])
            if who and until > at and beat.type == BeatType.HERO]
    return found, Screen(background=script.background, on=on, sung=sung)


def _alive(on: dict[str, Pos], drawings: dict[str, Drawing]) -> bool:
    return any(drawings[d].motion for d in on)


def _singer(on: dict[str, Pos], drawings: dict[str, Drawing]) -> str | None:
    """The drawing that sings just now: one with a mouth, big on screen, not a small stage of a row or ring."""
    return next((d for d, pos in on.items() if drawings[d].sings and pos not in KEPT_CLEAR), None)


def _apply(a: Action, beat: Beat, drawings: dict[str, Drawing], on: dict[str, Pos]) -> list[str]:
    """One action applied to what is on screen; returns what is wrong with it. The performer is one drawing
    at a time, or it looks like two creatures. Only the stages of a row or ring may be several of its drawings,
    as the stages of one life, and then none stands outside them."""
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
    row = [d for d, p in on.items() if p in KEPT_CLEAR and d != a.drawing]
    rest = [d for d, p in on.items() if p not in KEPT_CLEAR and d != a.drawing]
    taken = next((d for d in rest if on[d] == a.pos), None)
    if a.pos in KEPT_CLEAR:                                     # a row or ring needs its part of the screen to itself
        taken = next((d for d in rest if on[d] in KEPT_CLEAR[a.pos]), None) or next(
            (d for d in row if on[d] != a.pos), None)
    elif row and a.pos in KEPT_CLEAR[on[row[0]]]:
        taken = row[0]
    other = next((d for d in rest if drawings[d].kind == DrawingKind.PERFORMER), None)
    stage = next((d for d in row if drawings[d].kind == DrawingKind.PERFORMER), None)
    twice = (f"{what}: {other or stage} is on screen; the performer is one drawing at a time, or it looks like two "
             "creatures, so swap that one instead, or hide it first")
    spot = f"{what}: {a.pos} is taken by {taken}; hide or move it first" if taken else f"{what}: pos is needed"
    if a.verb == Verb.SHOW:
        if a.drawing in on:
            found.append(f"{what}: {a.drawing} is already on screen; move it instead")
        elif thing.kind == DrawingKind.PERFORMER and (other or stage and a.pos not in KEPT_CLEAR):
            found.append(twice)
        elif a.pos is None or taken:
            found.append(spot)
        elif a.pos in KEPT_CLEAR and len(row) == MAX_ROW:
            found.append(f"{what}: the {a.pos} is full; at most {MAX_ROW} stages in a {a.pos}")
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
        elif new.kind == DrawingKind.PERFORMER and (other or stage and on[a.drawing] not in KEPT_CLEAR):
            found.append(twice)
        else:
            on[a.to] = on.pop(a.drawing)
    elif a.verb == Verb.MOVE:
        if a.pos in KEPT_CLEAR:
            found.append(f"{what}: a drawing joins the {a.pos} with show, not move")
        elif a.pos is None or taken:
            found.append(spot)
        elif thing.kind == DrawingKind.PERFORMER and (other or stage):
            found.append(twice)
        else:
            on[a.drawing] = a.pos
    elif a.verb == Verb.ZOOM and (a.scale is None or not ZOOM_LEAST <= a.scale <= ZOOM_MOST[thing.kind]):
        found.append(f"{what}: scale must be between {ZOOM_LEAST} and {ZOOM_MOST[thing.kind]}; a {thing.kind} drawing "
                     "any bigger does not fit the screen, so start it smaller to leave room to grow")
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
        extra += [Action(verb=Verb.LIPSYNC, drawing=who, at=at, until=until) for who, at, until in screen.sung]
        out.append(BeatScript(beat=beat.id, background=script.background,
                              actions=sorted(script.actions + extra, key=lambda a: a.at),
                              start=beat.start, end=beat.end, type=beat.type))
    return out


def _text(verb: Verb, text: str, at: float | None, beat: Beat, duration: float) -> Action:
    at = beat.start if at is None else min(max(at, beat.start), beat.end)
    return Action(verb=verb, text=text, at=at, until=round(min(at + KEYWORD_SECONDS, duration), 3))
