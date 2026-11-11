import math

from pydantic import BaseModel, Field

from src.schema.storyboard.script_schema import AnimationScript, BeatScript, DrawingKind, Pos, Verb

# Where each position's centre sits, as shares of the frame's width and height, and how tall a drawing's
# square frame is drawn, as a share of the frame's height.
ANCHOR = {Pos.LEFT: (0.22, 0.54), Pos.CENTER: (0.5, 0.54), Pos.RIGHT: (0.78, 0.54), Pos.TOP: (0.5, 0.2),
          Pos.BOTTOM: (0.5, 0.8)}
SIZE = {DrawingKind.PERFORMER: 0.62, DrawingKind.OBJECT: 0.44, DrawingKind.ICON: 0.26}
# A row of stages shares out evenly, left to right, what the camera always shows of its scene, at ROW_Y down
# the frame. A stage's square fills ROW_FILL of its place across; the arrow to it from the stage before fills
# ARROW of the gap between the two subjects.
ROW_Y, ROW_FILL, ARROW = 0.54, 0.9, 0.6
# Seconds a drawing takes to pop in, to fade out, and to change into another; a move with no end takes MOVE.
IN, OUT, SWAP, MOVE = 0.35, 0.25, 0.3, 0.6
PULSE = 0.08                # how much a highlighted drawing grows at the top of each of its two pulses
# The camera: how far in the picture is when wide and when close, the closest it goes, and how far a pan
# slides, as a share of the frame's width. Wide is a little in, so a pan has picture to slide over.
WIDE, CLOSE, CLOSEST, PAN = 1.06, 1.2, 1.3, 0.025
EDGE = (1 - 1 / WIDE) / 2   # the strip along each side of a scene that the wide camera leaves out
BODY = 0.8                  # how much of its square a drawing's subject fills, more or less; the rest is clear
WORD = 0.25                 # seconds a key word takes to pop in and out
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
    it away, none when the scene ended with it on. slot: its centre's x when its place is the row. moves:
    (at, until, from, to); zooms: (at, until, from, to) scales; highlights: (at, until)."""

    drawing: str
    kind: DrawingKind
    born: float
    how: Verb
    pos: Pos
    scale: float = 1.0
    slot: float | None = None
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
    pan to); words: (at, until, text, centre x, centre y); arrows: (at, centre x, length across), between the
    stages of the row. view: the part of the scene the camera shows wherever it goes in it (left, top, right,
    bottom); every drawing is kept inside it, so no camera move cuts one off."""

    background: str
    start: float
    end: float
    zoom: float = WIDE
    tracks: list[Track] = Field(default_factory=list)
    camera: list[tuple[float, float, float, float, float, float]] = Field(default_factory=list)
    words: list[tuple[float, float, str, float, float]] = Field(default_factory=list)
    arrows: list[tuple[float, float, float]] = Field(default_factory=list)
    view: tuple[float, float, float, float] = (EDGE, EDGE, 1 - EDGE, 1 - EDGE)


class Item(BaseModel):
    """A drawing in one frame: its centre and its square's height as shares of the frame, before the camera;
    age: seconds since it appeared, which picks its loop's frame."""

    drawing: str
    x: float
    y: float
    size: float
    opacity: float
    age: float


class Frame(BaseModel):
    background: str
    zoom: float
    pan: float
    items: list[Item]
    words: list[tuple[str, float, float, float]]        # text, centre x, centre y, how far popped in (0-1)
    arrows: list[tuple[float, float, float, float]] = Field(default_factory=list)   # centre x, y, length, popped in


def scenes_of(board: AnimationScript, beats: list[BeatScript], wide: float) -> list[Scene]:
    """The script's beats as scenes of tracks, for a frame `wide` times as wide as it is tall. The script
    was checked when it was written, so an action on a drawing that is not on screen is skipped, not reported."""
    kinds, groups = {d.id: d.kind for d in board.drawings}, []
    for beat in beats:
        if groups and groups[-1][0].background == beat.background:
            groups[-1].append(beat)
        else:
            groups.append([beat])
    return [_scene(group, kinds, wide) for group in groups]


def _scene(beats: list[BeatScript], kinds: dict[str, DrawingKind], wide: float) -> Scene:
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
    reach, pans = 0.5 / max([scene.zoom, *(c[3] for c in scene.camera)]), [0.0, *(c[5] for c in scene.camera)]
    scene.view = (0.5 + max(pans) - reach, 0.5 - reach, 0.5 + min(pans) + reach, 0.5 + reach)
    on: dict[str, Track] = {}
    stages = sum(a.verb == Verb.SHOW and a.pos == Pos.ROW and a.drawing in kinds for a in actions)
    place, stood = (scene.view[2] - scene.view[0]) / max(1, stages), 0     # how wide one place in the row is; how many are taken
    for a in actions:
        track = on.get(a.drawing or "")
        if a.verb in (Verb.KEYWORD, Verb.NUMBER):
            scene.words.append((a.at, a.until or a.at + 2.0, a.text or "", 0.0, 0.0))
        elif a.verb == Verb.SHOW and a.pos and a.drawing in kinds:
            on[a.drawing] = Track(drawing=a.drawing, kind=kinds[a.drawing], born=a.at, how=Verb.SHOW, pos=a.pos)
            scene.tracks.append(on[a.drawing])
            if a.pos == Pos.ROW:
                on[a.drawing].slot = scene.view[0] + place * (stood + 0.5)
                on[a.drawing].scale = min(1.0, ROW_FILL * place * wide / SIZE[kinds[a.drawing]])
                if stood:
                    scene.arrows.append((a.at, scene.view[0] + place * stood, ARROW * (1 - ROW_FILL * BODY) * place))
                stood += 1
        elif track is None:
            continue
        elif a.verb == Verb.HIDE:
            on.pop(a.drawing).gone, track.out = a.at, Verb.HIDE
        elif a.verb == Verb.SWAP and a.to in kinds:
            on.pop(a.drawing).gone, track.out = a.at, Verb.SWAP
            on[a.to] = Track(drawing=a.to, kind=kinds[a.to], born=a.at, how=Verb.SWAP, pos=track.place(),
                             scale=track.size(), slot=track.slot)
            scene.tracks.append(on[a.to])
        elif a.verb == Verb.MOVE and a.pos:
            track.moves.append((a.at, a.until or a.at + MOVE, track.place(), a.pos))
        elif a.verb == Verb.ZOOM and a.scale:
            track.zooms.append((a.at, a.until or a.at + MOVE, track.size(), a.scale))
        elif a.verb == Verb.HIGHLIGHT:
            track.highlights.append((a.at, a.until or a.at + 1.0))
    for track in on.values():                                   # what is still on screen goes with the cut
        track.gone = scene.end
    for n, (at, until, text, _, _) in enumerate(scene.words):
        zoom, pan = _camera(scene, at)
        held = [_seen(_inside(*_stand(place, t.slot), size, wide, scene.view), size * BODY, zoom, pan, wide)
                for t in scene.tracks if t.born < until and t.gone > at
                for size in [SIZE[t.kind] * max([t.scale, *(z[3] for z in t.zooms)])]
                for place in (t.pos, *(m[3] for m in t.moves))]
        scene.words[n] = (at, until, text, *_spot(text, held, wide))
    return scene


def _spot(text: str, held: list[tuple[float, float, float, float]], wide: float) -> tuple[float, float]:
    """The spot where a word's plate covers the least of the boxes the drawings hold."""
    across = (len(text) * LETTER * WORD_PX + 2 * (WORD_PAD + WORD_RIM)) / wide
    w, h = min(across, WORD_WIDE), (WORD_PX + 2 * (WORD_PAD + WORD_RIM)) * min(1.0, WORD_WIDE / across)

    def covered(spot: tuple[float, float]) -> float:
        x, y = spot
        return sum(max(0.0, min(x + w / 2, right) - max(x - w / 2, left)) * max(0.0, min(y + h / 2, bottom) - max(y - h / 2, top))
                   for left, top, right, bottom in held)

    return min(WORD_SPOTS, key=covered)


def _stand(place: Pos, slot: float | None) -> tuple[float, float]:
    """Where a place's centre is; the row's places are worked out for each scene."""
    return (slot, ROW_Y) if place == Pos.ROW else ANCHOR[place]


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
    items = [item for track in scene.tracks if (item := _item(track, t, wide, scene.view)) is not None]
    order = {DrawingKind.OBJECT: 1, DrawingKind.PERFORMER: 2, DrawingKind.ICON: 3}
    kinds = {track.drawing: 0 if track.place() in (Pos.TOP, Pos.BOTTOM) and track.kind == DrawingKind.OBJECT
             else order[track.kind] for track in scene.tracks}
    words = [(text, x, y, min(1.0, (t - at) / WORD, (until - t) / WORD)) for at, until, text, x, y in scene.words
             if at <= t < until]
    arrows = [(x, ROW_Y, length, min(1.0, (t - at) / IN)) for at, x, length in scene.arrows if t >= at]
    return Frame(background=scene.background, zoom=zoom, pan=pan, items=sorted(items, key=lambda i: kinds[i.drawing]),
                 words=words, arrows=arrows)


def _item(track: Track, t: float, wide: float, view: tuple[float, float, float, float]) -> Item | None:
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
    x, y = _inside(x, y, SIZE[track.kind] * scale, wide, view)
    for at, until in track.highlights:
        if at <= t < until:
            scale *= 1 + PULSE * abs(math.sin(2 * math.pi * (t - at) / (until - at)))
    return Item(drawing=track.drawing, x=x, y=y, size=SIZE[track.kind] * scale * grow, opacity=opacity, age=t - track.born)


def _ease(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def _back(u: float) -> float:
    """Ease out past the end and settle, for a pop."""
    u = min(1.0, max(0.0, u)) - 1
    return 1 + 2.70158 * u ** 3 + 1.70158 * u ** 2
