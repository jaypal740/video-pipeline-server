import re
import tomllib

from src.schema.lesson.lesson_schema import BandRules, FormatRules, GradeBand, VideoType
from src.schema.storyboard.storyboard_schema import (Angle, CameraMove, CastKind, CastMember, Continuity, Direction,
                                                     Motion, Moves, Place, Position, Scale, ScreenDirection, Shot, ShotDraft,
                                                     ShotKind, ShotSize, Slot, Speed, StoryboardDraft, Transition)
from src.service.lesson.lesson_rules import TEMPLATES

HOST = "host"
# A subject may sit this many scale rungs away from its place (a ship on the open sea, a cable on the seabed).
SCALE_FIT = 2
# Word limits keep the image and video prompts built from these fields short.
LIMITS = {"start_state": 30, "action": 20, "end_state": 20, "cast look": 25, "place look": 30, "lighting": 12,
          "size": 10, "motif rule": 15, "shape": 40, "outfit": 20}
# A thing off screen this many shots in a row gets its reference sheet attached again when it returns.
REANCHOR_AFTER = 4

_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_QUOTE = re.compile(r"[\"“”]")
_DIGIT = re.compile(r"\d")
_NEGATIVE = re.compile(r"\b(no|not|without|never|none)\b", re.I)
_TEXT = re.compile(r"\b(text|labels?|labell?ed|captions?|charts?|graphs?|diagrams?|lettering|writing|written|"
                   r"subtitles?|logos?|signage|signposts?|banners?|posters?)\b", re.I)
# Only words that always mean a human-like face. "face" itself is left out (a prism, a dial and a cliff have
# faces), as are mouth and expression (river mouth, gene expression); "eye" next to a hyphen is "eye-level".
# A drawn face on a thing is caught for real by the keyframe stage's image check.
_FACE = re.compile(r"\b((?<!-)eyes?(?!-)|smil\w*|frown\w*|grin\w*|wink\w*|cartoon\w*|mascots?|cute)\b", re.I)
# A person in a shot's words. One who comes back in a later shot must be a cast member, or the keyframe stage
# has no reference to draw them from and they change from shot to shot; a passer-by, a crowd or a hand holding
# something is scenery and needs no cast entry.
_PERSON = re.compile(r"\b(girls?|boys?|men|man|women|woman|kids?|child(?:ren)?|teen(?:ager)?s?|students?|"
                     r"someone|somebody|persons?)\b", re.I)
_BRIDGES = {Transition.DISSOLVE, Transition.MATCH_CUT}
_ZOOMS = {CameraMove.PUSH_IN, CameraMove.PULL_BACK}

_SIZES = list(ShotSize)
_SCALES = list(Scale)
_SPEEDS = list(Speed)
_ESTABLISHING = {ShotSize.WIDE, ShotSize.MEDIUM_WIDE}
_SUNG_SIZES = {ShotSize.MEDIUM_CLOSE, ShotSize.CLOSE}
_NO_DIRECTION = {CameraMove.STATIC, CameraMove.PUSH_IN, CameraMove.PULL_BACK}
_DIRECTIONS = {CameraMove.PAN: {Direction.LEFT, Direction.RIGHT}, CameraMove.TRUCK: {Direction.LEFT, Direction.RIGHT},
               CameraMove.ORBIT: {Direction.LEFT, Direction.RIGHT}, CameraMove.TILT: {Direction.UP, Direction.DOWN},
               CameraMove.PEDESTAL: {Direction.UP, Direction.DOWN},
               CameraMove.FOLLOW: {Direction.LEFT, Direction.RIGHT, Direction.TOWARD, Direction.AWAY}}
_FLIPS = {(ScreenDirection.LEFT_TO_RIGHT, ScreenDirection.RIGHT_TO_LEFT),
          (ScreenDirection.RIGHT_TO_LEFT, ScreenDirection.LEFT_TO_RIGHT)}


def host_for(band: GradeBand) -> CastMember:
    """The band's fixed host, from src/templates/host/<band>.toml: the person (face, hair, body) without
    clothes; the clothes come from the wardrobe kit a video picks."""
    data = tomllib.loads((TEMPLATES / "host" / f"{band}.toml").read_text())
    return CastMember(id=HOST, kind=CastKind.PERSON, look=data["look"], size=data["size"], scale=Scale.HUMAN)


def wardrobe_for(band: GradeBand) -> dict[str, str]:
    """The band's outfit kits by name."""
    return tomllib.loads((TEMPLATES / "host" / f"{band}.toml").read_text())["wardrobe"]


def outfit_text(band: GradeBand, outfit: str) -> str:
    """An outfit as words: a wardrobe kit name expands to its description, anything else is used as is."""
    return wardrobe_for(band).get(outfit, outfit).rstrip(".")


def host_in(band: GradeBand, outfit: str) -> CastMember:
    """The host dressed in one outfit (a kit name or free words), as drawn."""
    host = host_for(band)
    return host.model_copy(update={"look": f"{host.look}, wearing {outfit_text(band, outfit)}"})


def _words(field: str, text: str, limit: int, faces_allowed: bool, digits_allowed: bool = False) -> list[str]:
    """Word rules for one field. Digits are banned only in text that reaches an image prompt; a real size
    ("about 90 metres") and the shape sentence never do."""
    found = []
    if len(text.split()) > limit:
        found.append(f"{field} has {len(text.split())} words; at most {limit}")
    if _QUOTE.search(text):
        found.append(f"{field} has quotation marks; image models draw quoted words as text")
    if not digits_allowed and _DIGIT.search(text):
        found.append(f"{field} has digits; our code draws every number, so leave figures out of the picture")
    if m := _NEGATIVE.search(text):
        found.append(f"{field} says {m.group(0)!r}; describe what is there instead, models draw what they read")
    if m := _TEXT.search(text):
        found.append(f"{field} mentions {m.group(0)!r}; pictures have no text, our code adds it")
    if not faces_allowed and (m := _FACE.search(text)):
        found.append(f"{field} mentions {m.group(0)!r}; only people have faces and expressions")
    return found


def wanted(slots: list[Slot]) -> set[str]:
    """The slot ids the model writes shots for: every slot that is not a repeat."""
    return {s.id for s in slots if s.repeat_of is None}


def header_problems(draft: StoryboardDraft, band: BandRules, slots: list[Slot]) -> list[str]:
    """Problems with the cast, places, motifs, shape and outfit; any of these re-asks the whole board."""
    found = _words("shape", draft.shape, LIMITS["shape"], faces_allowed=True, digits_allowed=True)
    wardrobe = wardrobe_for(band.band)
    for p in draft.places:
        if p.host_outfit and p.host_outfit not in wardrobe:
            found += _words(f"place {p.id} host_outfit", p.host_outfit, LIMITS["outfit"], faces_allowed=True)
    ids = [c.id for c in draft.cast] + [p.id for p in draft.places]
    for bad in sorted({i for i in ids if not _ID.match(i) or ids.count(i) > 1}):
        found.append(f"id {bad!r} must be unique, lowercase letters, digits and underscores")
    if HOST in ids:
        found.append(f"{HOST!r} is given; do not declare it in cast or places")
    for c in draft.cast:
        found += _words(f"cast {c.id} look", c.look, LIMITS["cast look"], faces_allowed=c.kind == CastKind.PERSON)
        found += _words(f"cast {c.id} size", c.size, LIMITS["size"], faces_allowed=True, digits_allowed=True)
    for p in draft.places:
        found += _words(f"place {p.id} look", p.look, LIMITS["place look"], faces_allowed=False)
        found += _words(f"place {p.id} lighting", p.lighting, LIMITS["lighting"], faces_allowed=False)
    for m in draft.motifs:
        found += _words(f"motif {m.thing} rule", m.rule, LIMITS["motif rule"], faces_allowed=False)
    if not draft.places:
        found.append("places must list at least one place")
    used = {s.place for s in draft.shots if s.slot in wanted(slots)}       # the given host place is not counted
    in_use = sum(p.id in used for p in draft.places)
    if in_use > band.max_places:
        found.append(f"{in_use} places used; at most {band.max_places} for grades {band.band}")
    return found


def fix(shot: ShotDraft, slot: Slot, band: BandRules, people: set[str]) -> list[str]:
    """What code corrects itself instead of re-asking: camera details, and the framing sung and number
    shots need. A sung shot shows one face for lip-sync: the host, with at most one thing. Returns what
    changed, for the log."""
    before = shot.model_dump(mode="json")
    cam = shot.camera
    if cam.move not in band.camera_moves:
        cam.move = CameraMove.STATIC
    if slot.kind == ShotKind.SUNG:
        shot.subjects = [HOST] + [s for s in shot.subjects if s != HOST and s not in people][:1]
        shot.angle, shot.position, shot.motion = Angle.EYE, Position.CENTRE, Motion.LOW
        shot.screen_direction = ScreenDirection.NONE
        shot.moves = Moves.SUBJECT                              # the host sings; lip-sync needs the face
        if shot.size not in _SUNG_SIZES:
            shot.size = ShotSize.MEDIUM_CLOSE
        if cam.move not in (CameraMove.STATIC, CameraMove.PUSH_IN):
            cam.move = CameraMove.STATIC
    if slot.kind == ShotKind.NUMBER and shot.position == Position.CENTRE:
        shot.position = Position.RIGHT_THIRD
    if cam.move in _NO_DIRECTION:
        cam.direction = None
    if cam.move == CameraMove.STATIC:
        cam.speed = None
    else:
        cam.speed = cam.speed or Speed.SLOW
        if cam.speed == Speed.FAST and shot.motion == Motion.HIGH:
            cam.speed = Speed.SLOW
        limit = _SPEEDS.index(Speed(band.max_camera_speed))
        if _SPEEDS.index(cam.speed) > limit:
            cam.speed = _SPEEDS[limit]
    after = shot.model_dump(mode="json")
    return [f"{slot.id}: {field} {before[field]!r} -> {after[field]!r}" for field in after if before[field] != after[field]]


def timeline(draft: StoryboardDraft, slots: list[Slot]) -> list[tuple[Slot, ShotDraft | None]]:
    """Every slot with the shot it shows; a repeat shows the shot of the slot it copies."""
    by_slot = {s.slot: s for s in draft.shots}
    return [(slot, by_slot.get(slot.repeat_of or slot.id)) for slot in slots]


def shot_problems(draft: StoryboardDraft, slots: list[Slot], band: BandRules, fmt: FormatRules,
                  host: CastMember) -> dict[str, list[str]]:
    """Problems per slot id; only those shots are re-asked."""
    found: dict[str, list[str]] = {}

    def add(slot_id: str, problem: str) -> None:
        found.setdefault(slot_id, []).append(problem)

    cast = {c.id: c for c in draft.cast + [host]}
    places = {p.id: p for p in draft.places}
    written = [s.slot for s in draft.shots]
    for slot_id in sorted(wanted(slots) - set(written)):
        add(slot_id, "this slot has no shot")
    for slot_id in sorted({s for s in written if written.count(s) > 1}):
        add(slot_id, "this slot has more than one shot; give exactly one")

    unlisted: dict[str, list[str]] = {}
    for slot in slots:
        shot = next((s for s in draft.shots if s.slot == slot.id), None)
        if slot.repeat_of is not None or shot is None:
            continue
        if shot.place not in places:
            add(slot.id, f"place {shot.place!r} is not in places")
        elif slot.kind == ShotKind.SUNG and not places[shot.place].walkable:
            add(slot.id, f"a sung shot needs a place where a person can really stand; {shot.place!r} is not walkable")
        unknown = [s for s in shot.subjects if s not in cast]
        if unknown:
            add(slot.id, f"subjects {unknown} are not in cast")
        if len(shot.subjects) > 3:
            add(slot.id, "give at most 3 subjects")
        if shot.place in places:
            for s in shot.subjects:
                if s in cast and abs(_SCALES.index(cast[s].scale) - _SCALES.index(places[shot.place].scale)) > SCALE_FIT:
                    add(slot.id, f"subject {s!r} ({cast[s].scale}) is far too small or large for place "
                                 f"{shot.place!r} ({places[shot.place].scale}); show it in a place at its own "
                                 "scale, or leave subjects empty and let the place carry the shot")
        people = sum(cast[s].kind == CastKind.PERSON for s in shot.subjects if s in cast)
        if people > band.max_people_in_frame:
            add(slot.id, f"{people} people in frame; at most {band.max_people_in_frame} for grades {band.band}")
        if HOST in shot.subjects and shot.place in places and not places[shot.place].host_outfit:
            add(slot.id, f"the host appears in {shot.place!r}, so that place needs a host_outfit")
        if HOST in shot.subjects and slot.kind != ShotKind.SUNG and fmt.type != VideoType.PERSONALISED_HERO:
            add(slot.id, f"the host appears only in sung shots in a {fmt.type} video")
        move, direction = shot.camera.move, shot.camera.direction
        if shot.moves == Moves.CAMERA and move == CameraMove.STATIC:
            add(slot.id, "moves is camera but the camera is static; give the camera a move, or set moves to subject")
        if move in _DIRECTIONS and direction not in _DIRECTIONS[move]:
            add(slot.id, f"camera {move} needs a direction from {sorted(_DIRECTIONS[move])}")
        for field in ("start_state", "action", "end_state"):
            for p in _words(field, getattr(shot, field), LIMITS[field], faces_allowed=people > 0):
                add(slot.id, p)
        if people == 0:
            for m in _PERSON.finditer(f"{shot.start_state} {shot.action} {shot.end_state}"):
                unlisted.setdefault(m.group(0).lower().rstrip("s"), []).append(slot.id)
    for word, where in unlisted.items():                        # the same person word in two or more shots
        if len(seen := sorted(set(where))) > 1:
            for slot_id in seen:
                add(slot_id, f"a {word!r} appears in {', '.join(seen)} but is not a cast member; declare them in cast "
                             f"with kind person and list them in those shots if they are the same person, or make one "
                             "of them a crowd or a passer-by")

    # Shot-to-shot rules. A repeat copies a shot already checked in its own context, so it is never
    # blamed; the shot after it is checked against it like any other.
    pairs = timeline(draft, slots)
    established: set[str] = set()
    for (prev_slot, prev), (slot, shot) in zip([(None, None)] + pairs, pairs):
        if shot is None or slot.repeat_of is not None:
            continue
        new_place = prev is None or prev.place != shot.place
        if prev is not None and new_place and set(prev_slot.lines) & set(slot.lines):
            add(slot.id, "a lyric line is split across two places; keep one line in one place")
        if slot.kind != ShotKind.SUNG and shot.place not in established:
            if shot.size not in _ESTABLISHING:
                add(slot.id, f"first picture shot in place {shot.place!r} must be wide or medium_wide to show "
                             "where we are")
            established.add(shot.place)
        if prev is None or slot.kind == ShotKind.SUNG or prev_slot.kind == ShotKind.SUNG:
            continue
        if not new_place:
            jump = abs(_SIZES.index(shot.size) - _SIZES.index(prev.size))
            if jump > band.max_size_jump:
                add(slot.id, f"shot size jumps {jump} steps from {prev.size} to {shot.size}; at most "
                             f"{band.max_size_jump} within a place")
            if (prev.screen_direction, shot.screen_direction) in _FLIPS:
                add(slot.id, f"screen direction flips from {prev.screen_direction} to {shot.screen_direction} in "
                             "the same place; keep it or put a neutral shot between")
        elif fmt.max_scale_step is not None and prev.place in places and shot.place in places:
            step = abs(_SCALES.index(places[shot.place].scale) - _SCALES.index(places[prev.place].scale))
            bridged = shot.transition_in in _BRIDGES or prev.camera.move in _ZOOMS
            if step > fmt.max_scale_step and not bridged:
                add(slot.id, f"scale jumps {step} steps from {places[prev.place].scale} to "
                             f"{places[shot.place].scale} on a plain cut; bring the viewer with it: come in on a "
                             "dissolve or match_cut, or let the shot before push in or pull back")
    return found


def assemble(draft: StoryboardDraft, slots: list[Slot], host: CastMember) -> tuple[list[Shot], list[str]]:
    """The final shots in time order, with what code decides: performer, continuity, repeats copied;
    plus flags for things returning after a long absence."""
    things = {c.id for c in draft.cast if c.kind == CastKind.THING}
    shots, flags, last_seen = [], [], {}
    for i, (slot, draft_shot) in enumerate(timeline(draft, slots)):
        if draft_shot is None:
            raise ValueError(f"slot {slot.id} has no shot to assemble; check the board before assembling it")
        prev = shots[-1] if shots else None
        if prev is None:
            continuity = Continuity.NEW_SCENE
        elif prev.place == draft_shot.place:
            continuity = Continuity.SAME_PLACE
        elif set(prev.subjects) & set(draft_shot.subjects):
            continuity = Continuity.SAME_SUBJECT
        else:
            continuity = Continuity.NEW_SCENE
        shots.append(Shot(**draft_shot.model_dump(exclude={"slot"}), slot=slot.id, id=slot.id, start=slot.start,
                          end=slot.end, lines=slot.lines, kind=slot.kind, number_text=slot.number_text,
                          repeat_of=slot.repeat_of, performer=HOST if slot.kind == ShotKind.SUNG else None,
                          continuity=continuity))
        for subject in draft_shot.subjects:
            if subject in things and subject in last_seen and i - last_seen[subject] - 1 >= REANCHOR_AFTER:
                flags.append(f"{slot.id}: {subject} returns after {i - last_seen[subject] - 1} shots; attach its "
                             "reference sheet")
            last_seen[subject] = i
    return shots, flags
