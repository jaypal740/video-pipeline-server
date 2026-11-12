import math

from pydantic import BaseModel, Field

from src.schema.storyboard.script_schema import AnimationScript, BeatScript, DrawingKind, DrawingList, Pos, Verb

# Where each position's centre sits, as shares of the frame's width and height, and how tall a drawing's
# square frame is drawn, as a share of the frame's height.
ANCHOR = {Pos.LEFT: (0.22, 0.54), Pos.CENTER: (0.5, 0.54), Pos.RIGHT: (0.78, 0.54), Pos.TOP: (0.5, 0.2),
          Pos.BOTTOM: (0.5, 0.8)}
SIZE = {DrawingKind.PERFORMER: 0.62, DrawingKind.OBJECT: 0.44, DrawingKind.ICON: 0.26}
# A row of stages shares out evenly, left to right, what the camera shows all the while the row is up, at ROW_Y
# down the frame. A stage's square fills ROW_FILL of its place across; the arrow to it from the stage before fills
# ARROW of the gap between the two subjects.
ROW_Y, ROW_FILL, ARROW = 0.54, 0.9, 0.6
# A ring of stages goes round that view, clockwise from the top, with an arrow from each stage to the next
# and from the last back to the first. A stage's square is RING_FILL of the view's height, the hole in the
# middle holds one drawing at center, made to fit and stood in the middle of it, and no ring arrow is longer
# than RING_ARROW of the width.
RING_FILL, RING_ARROW = 0.32, 0.08
# A stage that has a name is drawn NAMED of its size, moved up in its square, and code writes the name in the
# room below it, in letters NAME_PX of the frame's height.
NAMED, NAME_PX = 0.85, 0.04
LAID = (Pos.ROW, Pos.RING)  # the places code lays out for each scene
# Seconds a drawing takes to pop in, to fade out, and to change into another; a move with no end takes MOVE.
IN, OUT, SWAP, MOVE = 0.35, 0.25, 0.3, 0.6
PULSE = 0.08                # how much a highlighted drawing grows at the top of each of its two pulses
# The camera: how far in the picture is when wide and when close, the closest it goes, and how far a pan
# slides, as a share of the frame's width. Wide is a little in, so a pan has picture to slide over.
WIDE, CLOSE, CLOSEST, PAN = 1.06, 1.2, 1.3, 0.025
EDGE = (1 - 1 / WIDE) / 2   # the strip along each side of a scene that the wide camera leaves out
BODY = 0.8                  # how much of its square a drawing's subject fills, more or less; the rest is clear
WORD = 0.25                 # seconds a key word takes to pop in and out
# One drawing hides another when their subjects share more than COVER of the smaller one, for HIDDEN seconds or
# longer; drawings that only pass each other do not. `covers` looks every LOOK seconds.
COVER, HIDDEN, LOOK = 0.2, 1.0, 0.25
# A key word's plate, as shares of the frame's height: its letters, the room around them and its rim. LETTER
# is a letter's width as a share of its height, near enough to choose a spot by; the painter measures the
# real one. No plate is wider than WORD_WIDE of the frame's width: a longer word is drawn smaller, and is
# moved in from the frame's side to leave WORD_GAP of the width clear.
WORD_PX, WORD_PAD, WORD_RIM, LETTER, WORD_WIDE, WORD_GAP = 0.1, 0.035, 0.008, 0.6, 0.42, 0.02
# Where a key word's centre can go. It takes the spot where its plate covers the least of the drawings on
# screen while it shows, the earliest of these when several are as clear, so a word never covers what it names.
WORD_SPOTS = ((0.5, 0.13), (0.22, 0.3), (0.78, 0.3), (0.5, 0.72), (0.22, 0.13), (0.78, 0.13))


class Track(BaseModel):
    """One drawing's stay on screen. how: the verb that brought it (show or swap); out: the verb that took
    it away, none when the scene ended with it on. slot: its centre when its place is in a row or ring. view:
    the part of the scene the camera shows all the while it is on screen (left, top, right, bottom); it is kept
    inside it, so no camera move cuts it off. name: the stage's name, written under it while it stands in a row
    or ring. moves: (at, until, from, to); zooms: (at, until, from, to) scales; highlights: (at, until)."""

    drawing: str
    kind: DrawingKind
    born: float
    how: Verb
    pos: Pos
    name: str | None = None
    scale: float = 1.0
    slot: tuple[float, float] | None = None
    view: tuple[float, float, float, float] = (EDGE, EDGE, 1 - EDGE, 1 - EDGE)
    gone: float | None = None
    out: Verb | None = None
    moves: list[tuple[float, float, Pos, Pos]] = Field(default_factory=list)
    zooms: list[tuple[float, float, float, float]] = Field(default_factory=list)
    highlights: list[tuple[float, float]] = Field(default_factory=list)

    def place(self) -> Pos:
        return self.moves[-1][3] if self.moves else self.pos

    def size(self) -> float:
        return self.zooms[-1][3] if self.zooms else self.scale


class Scene(BaseModel):
    """The beats in a row that share one background: drawings and the camera carry on through them, and a
    new background starts a new scene with a clear screen. camera: (at, until, zoom from, zoom to, pan from,
    pan to); words: (at, until, text, centre x, centre y); arrows: (at, until, centre x, centre y, length
    across, degrees turned clockwise from pointing right), between the stages of a row or ring, until none
    when they stay to the scene's end."""

    background: str
    start: float
    end: float
    zoom: float = WIDE
    tracks: list[Track] = Field(default_factory=list)
    camera: list[tuple[float, float, float, float, float, float]] = Field(default_factory=list)
    words: list[tuple[float, float, str, float, float]] = Field(default_factory=list)
    arrows: list[tuple[float, float | None, float, float, float, float]] = Field(default_factory=list)


class Item(BaseModel):
    """A drawing in one frame: its centre and its square's height as shares of the frame, before the camera;
    age: seconds since it appeared, which picks its loop's frame; name: what code writes under it."""

    drawing: str
    x: float
    y: float
    size: float
    opacity: float
    age: float
    name: str | None = None


class Frame(BaseModel):
    background: str
    zoom: float
    pan: float
    items: list[Item]
    words: list[tuple[str, float, float, float]]        # text, centre x, centre y, how far popped in (0-1)
    arrows: list[tuple[float, float, float, float, float]] = Field(default_factory=list)   # centre x, y, length, turn, popped in


def scenes_of(board: AnimationScript | DrawingList, beats: list[BeatScript], wide: float) -> list[Scene]:
    """The script's beats as scenes of tracks, for a frame `wide` times as wide as it is tall. The script
    was checked when it was written, so an action on a drawing that is not on screen is skipped, not reported."""
    kinds, names, groups = {d.id: d.kind for d in board.drawings}, {d.id: d.name for d in board.drawings}, []
    for beat in beats:
        if groups and groups[-1][0].background == beat.background:
            groups[-1].append(beat)
        else:
            groups.append([beat])
    return [_scene(group, kinds, names, wide) for group in groups]


def _scene(beats: list[BeatScript], kinds: dict[str, DrawingKind], names: dict[str, str | None], wide: float) -> Scene:
    actions = sorted((a for b in beats for a in b.actions), key=lambda a: a.at)
    first = next((a.verb for a in actions if a.verb in (Verb.PUSH_IN, Verb.PULL_BACK)), None)
    scene = Scene(background=beats[0].background, start=beats[0].start, end=beats[-1].end,
                  zoom=CLOSE if first == Verb.PULL_BACK else WIDE)                    # a pull back needs to start close
    for a in actions:
        if a.verb in (Verb.PUSH_IN, Verb.PULL_BACK, Verb.PAN):
            zoom, pan = (scene.camera[-1][3], scene.camera[-1][5]) if scene.camera else (scene.zoom, 0.0)
            to = {Verb.PUSH_IN: min(CLOSEST, CLOSE if zoom < CLOSE else zoom + 0.1), Verb.PULL_BACK: WIDE}.get(a.verb, zoom)
            slide = pan + (PAN if a.pos == Pos.RIGHT else -PAN) if a.verb == Verb.PAN else pan
            scene.camera.append((a.at, a.until or a.at + MOVE, zoom, to, pan, slide))
    on: dict[str, Track] = {}
    laid, stays = _laid(actions, kinds), []         # stays: each drawing shown, with those swapped in after it
    for i, a in enumerate(actions):
        track = on.get(a.drawing or "")
        if a.verb in (Verb.KEYWORD, Verb.NUMBER):
            scene.words.append((a.at, a.until or a.at + 2.0, a.text or "", 0.0, 0.0))
        elif a.verb == Verb.SHOW and a.pos and a.drawing in kinds:
            on[a.drawing] = Track(drawing=a.drawing, kind=kinds[a.drawing], born=a.at, how=Verb.SHOW, pos=a.pos,
                                  name=names[a.drawing] if a.pos in LAID else None)
            scene.tracks.append(on[a.drawing])
            stays.append([on[a.drawing]])
            (n, stages, since, until), fit, arrows = laid.get(i, (0, 1, a.at, None)), None, []
            if a.pos in LAID:                                               # what the camera shows while the row or ring is up
                on[a.drawing].view = view = _view(scene, since, scene.end if until is None else until)
                left, top, right, bottom = view
            if a.pos == Pos.ROW:
                place = (right - left) / stages                             # how wide one place in the row is
                on[a.drawing].slot, fit = (left + place * (n + 0.5), ROW_Y), ROW_FILL * place * wide
                if n:
                    arrows.append((left + place * n, ROW_Y, ARROW * (1 - ROW_FILL * BODY) * place, 0.0))
            elif a.pos == Pos.RING:
                side = RING_FILL * (bottom - top)                           # a ring stage's square
                on[a.drawing].slot, fit = _round(view, side, wide, n / stages), side
                for k in [n] * bool(n) + [n + 1] * (n and n + 1 == stages):                  # to this stage; from the last, back to the first
                    (x0, y0), (x1, y1) = (_round(view, side, wide, j / stages) for j in (k - 1, k))
                    gap = math.hypot((x1 - x0) * wide, y1 - y0) - side * BODY
                    arrows.append(((x0 + x1) / 2, (y0 + y1) / 2, min(RING_ARROW, ARROW * gap / wide),
                                   math.degrees(math.atan2(y1 - y0, (x1 - x0) * wide))))
            if fit:
                on[a.drawing].scale = min(1.0, fit / SIZE[kinds[a.drawing]])
            scene.arrows += [(a.at, until, *arrow) for arrow in arrows]
        elif track is None:
            continue
        elif a.verb == Verb.HIDE:
            on.pop(a.drawing).gone, track.out = a.at, Verb.HIDE
        elif a.verb == Verb.SWAP and a.to in kinds:
            on.pop(a.drawing).gone, track.out = a.at, Verb.SWAP
            on[a.to] = Track(drawing=a.to, kind=kinds[a.to], born=a.at, how=Verb.SWAP, pos=track.place(),
                             scale=track.size(), slot=track.slot, view=track.view,
                             name=names[a.to] if track.place() in LAID else None)
            scene.tracks.append(on[a.to])
            next(stay for stay in stays if stay[-1] is track).append(on[a.to])
        elif a.verb == Verb.MOVE and a.pos:
            track.moves.append((a.at, a.until or a.at + MOVE, track.place(), a.pos))
        elif a.verb == Verb.ZOOM and a.scale:
            track.zooms.append((a.at, a.until or a.at + MOVE, track.size(), a.scale))
        elif a.verb == Verb.HIGHLIGHT:
            track.highlights.append((a.at, a.until or a.at + 1.0))
    for track in on.values():                                   # what is still on screen goes with the cut
        track.gone = scene.end
    for stay in stays:                                          # a drawing and those swapped in after it share one view
        if stay[0].slot is None:
            view = _view(scene, stay[0].born, stay[-1].gone)
            for track in stay:
                track.view = view
    rings = [t for t in scene.tracks if t.pos == Pos.RING]
    for track in scene.tracks:                                  # what stands in a ring's middle fits its hole, in the middle of it
        ring = next((r for r in rings if track.born < r.gone and r.born < track.gone), None)
        if track.pos == Pos.CENTER and ring:
            left, top, right, bottom = ring.view
            side = RING_FILL * (bottom - top)
            track.view = (left, top + side, right, bottom - side)
            fit = (bottom - top - 2 * side) / SIZE[track.kind]
            track.scale = min(track.scale, fit)
            track.zooms = [(at, until, min(start, fit), min(end, fit)) for at, until, start, end in track.zooms]
    for n, (at, until, text, _, _) in enumerate(scene.words):
        zoom, pan = _camera(scene, at)
        held = [_seen(_inside(*_stand(place, t.slot), size, wide, t.view), size * BODY, zoom, pan, wide)
                for t in scene.tracks if t.born < until and t.gone > at
                for size in [SIZE[t.kind] * max([t.scale, *(z[3] for z in t.zooms)])]
                for place in (t.pos, *(m[3] for m in t.moves))]
        held += [_seen((x, y), length * wide, zoom, pan, wide) for since, to, x, y, length, _ in scene.arrows
                 if since < until and (to is None or to > at)]                # a word covers no arrow either
        scene.words[n] = (at, until, text, *_spot(text, held, wide))
    return scene


def _laid(actions: list, kinds: dict[str, DrawingKind]) -> dict[int, tuple[int, int, float, float | None]]:
    """For each show at a row or ring, by its place in `actions`: which stage it is, how many stages its row
    or ring has, when that row or ring began and when it is over (none if it lasts to the scene's end). A row
    or ring is over when its last stage is hidden or moved out of it, and the next one built in the scene starts
    afresh."""
    on, built, out = {}, {}, {}
    for i, a in enumerate(actions):
        if a.verb == Verb.SHOW and a.pos in LAID and a.drawing in kinds:
            on[a.drawing] = a.pos
            built.setdefault(a.pos, []).append(i)
        elif a.verb == Verb.SWAP and a.drawing in on and a.to in kinds:
            on[a.to] = on.pop(a.drawing)
        elif a.verb in (Verb.HIDE, Verb.MOVE) and a.drawing in on:
            pos = on.pop(a.drawing)
            if pos not in on.values():
                shown = built.pop(pos)
                out.update({j: (n, len(shown), actions[shown[0]].at, a.at) for n, j in enumerate(shown)})
    out.update({j: (n, len(shown), actions[shown[0]].at, None) for shown in built.values() for n, j in enumerate(shown)})
    return out


def _spot(text: str, held: list[tuple[float, float, float, float]], wide: float) -> tuple[float, float]:
    """The spot where a word's plate covers the least of the boxes the drawings hold."""
    across = (len(text) * LETTER * WORD_PX + 2 * (WORD_PAD + WORD_RIM)) / wide
    w, h = min(across, WORD_WIDE), (WORD_PX + 2 * (WORD_PAD + WORD_RIM)) * min(1.0, WORD_WIDE / across)

    def covered(spot: tuple[float, float]) -> float:
        x, y = spot
        return sum(max(0.0, min(x + w / 2, right) - max(x - w / 2, left)) * max(0.0, min(y + h / 2, bottom) - max(y - h / 2, top))
                   for left, top, right, bottom in held)

    return min(WORD_SPOTS, key=covered)


def _stand(place: Pos, slot: tuple[float, float] | None) -> tuple[float, float]:
    """Where a place's centre is; the places of a row or ring are worked out for each scene."""
    return slot if place in LAID else ANCHOR[place]


def _round(view: tuple[float, float, float, float], side: float, wide: float, turn: float) -> tuple[float, float]:
    """The point `turn` of the way round the ring, clockwise from the top (0 to 1): where a stage's centre
    sits so that its square, `side` tall, stays inside the view."""
    left, top, right, bottom = view
    angle = 2 * math.pi * turn
    return ((left + right) / 2 + ((right - left) / 2 - side / wide / 2) * math.sin(angle),
            (top + bottom) / 2 - ((bottom - top) / 2 - side / 2) * math.cos(angle))


def _view(scene: Scene, at: float, until: float) -> tuple[float, float, float, float]:
    """The part of the scene the camera shows at every moment from `at` to `until`: left, top, right, bottom."""
    zooms, pans = zip(*(_camera(scene, t) for t in (at, until, *(t for c in scene.camera for t in c[:2] if at < t < until))))
    reach = 0.5 / max(zooms)
    return 0.5 + max(pans) - reach, 0.5 - reach, 0.5 + min(pans) + reach, 0.5 + reach


def _inside(x: float, y: float, size: float, wide: float, view: tuple[float, float, float, float]) -> tuple[float, float]:
    """A drawing's centre, moved so that its whole square is in the view; a square too big for the view sits
    in the middle of it."""
    across, down = size / wide / 2, size / 2
    left, top, right, bottom = view
    return (min(max(x, left + across), right - across) if 2 * across < right - left else (left + right) / 2,
            min(max(y, top + down), bottom - down) if 2 * down < bottom - top else (top + bottom) / 2)


def _seen(centre: tuple[float, float], size: float, zoom: float, pan: float, wide: float) -> tuple[float, float, float, float]:
    """The box a square of that size takes up in the frame through the camera: left, top, right, bottom."""
    x, y = 0.5 + (centre[0] - 0.5 - pan) * zoom, 0.5 + (centre[1] - 0.5) * zoom
    across, down = size * zoom / wide / 2, size * zoom / 2
    return x - across, y - down, x + across, y + down


def _camera(scene: Scene, t: float) -> tuple[float, float]:
    """How far in the camera is and how far it has slid."""
    zoom, pan = scene.zoom, 0.0
    for at, until, z0, z1, p0, p1 in scene.camera:
        if t >= at:
            u = _ease((t - at) / (until - at))
            zoom, pan = z0 + (z1 - z0) * u, p0 + (p1 - p0) * u
    return zoom, pan


def frame_at(scenes: list[Scene], t: float, wide: float) -> Frame:
    """What one moment of the film shows."""
    scene = next((s for s in scenes if s.start <= t < s.end), scenes[-1])
    zoom, pan = _camera(scene, t)
    items = [item for track in scene.tracks if (item := _item(track, t, wide)) is not None]
    order = {DrawingKind.OBJECT: 1, DrawingKind.PERFORMER: 2, DrawingKind.ICON: 3}
    kinds = {track.drawing: 0 if track.place() in (Pos.TOP, Pos.BOTTOM) and track.kind == DrawingKind.OBJECT
             else order[track.kind] for track in scene.tracks}
    words = [(text, x, y, min(1.0, (t - at) / WORD, (until - t) / WORD)) for at, until, text, x, y in scene.words
             if at <= t < until]
    arrows = [(x, y, length, turn, min(1.0, (t - at) / IN)) for at, until, x, y, length, turn in scene.arrows
              if at <= t and (until is None or t < until)]
    return Frame(background=scene.background, zoom=zoom, pan=pan, items=sorted(items, key=lambda i: kinds[i.drawing]),
                 words=words, arrows=arrows)


def _item(track: Track, t: float, wide: float) -> Item | None:
    """A track at one moment, or nothing when it is not on screen then."""
    leaving = {Verb.HIDE: OUT, Verb.SWAP: SWAP}.get(track.out, 0.0)
    if t < track.born or (track.gone is not None and t >= track.gone + leaving):
        return None
    arriving = (t - track.born) / (IN if track.how == Verb.SHOW else SWAP)
    opacity, grow = (min(1.0, 3 * arriving), 0.5 + 0.5 * _back(arriving)) if track.how == Verb.SHOW else (
        _ease(arriving), 0.9 + 0.1 * _ease(arriving))
    if track.gone is not None and t >= track.gone:
        left = _ease((t - track.gone) / leaving)
        opacity, grow = opacity * (1 - left), grow * (1 - 0.2 * left if track.out == Verb.HIDE else 1)
    (x, y), scale = _stand(track.pos, track.slot), track.scale
    for at, until, start, end in track.moves:
        if t >= at:
            u = _ease((t - at) / (until - at))
            x, y = (a + (b - a) * u for a, b in zip(_stand(start, track.slot), _stand(end, track.slot)))
    for at, until, start, end in track.zooms:
        if t >= at:
            scale = start + (end - start) * _ease((t - at) / (until - at))
    x, y = _inside(x, y, SIZE[track.kind] * scale, wide, track.view)
    for at, until in track.highlights:
        if at <= t < until:
            scale *= 1 + PULSE * abs(math.sin(2 * math.pi * (t - at) / (until - at)))
    size = SIZE[track.kind] * scale * grow
    name = None if any(t >= at for at, *_ in track.moves) else track.name    # a stage moved out of its row or ring is no longer named
    if name:                                                                # room below it for its name
        y, size = y - size * (1 - NAMED) / 2, size * NAMED
    return Item(drawing=track.drawing, x=x, y=y, size=size, opacity=opacity, age=t - track.born, name=name)


def covers(scenes: list[Scene], wide: float) -> list[tuple[float, str, str]]:
    """Where one drawing hides much of another: (from when, the drawing in front, the one behind), once for
    each time a pair overlaps."""
    found = []
    for scene in scenes:
        since: dict[tuple[str, str], int] = {}                  # each overlapping pair, and the look it began at
        for n in range(int((scene.end - scene.start) / LOOK)):
            items = [i for i in frame_at([scene], scene.start + n * LOOK, wide).items if i.opacity == 1]
            boxes = [(i.x - i.size * BODY / wide / 2, i.y - i.size * BODY / 2, i.x + i.size * BODY / wide / 2,
                      i.y + i.size * BODY / 2) for i in items]
            now = {(front.drawing, behind.drawing) for k, (behind, a) in enumerate(zip(items, boxes))
                   for front, b in list(zip(items, boxes))[k + 1:]
                   if max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
                   > COVER * min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))}
            since = {pair: since.get(pair, n) for pair in now}
            found += [(scene.start + first * LOOK, *pair) for pair, first in since.items() if n - first == round(HIDDEN / LOOK)]
    return found


def _ease(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def _back(u: float) -> float:
    """Ease out past the end and settle, for a pop."""
    u = min(1.0, max(0.0, u)) - 1
    return 1 + 2.70158 * u ** 3 + 1.70158 * u ** 2
