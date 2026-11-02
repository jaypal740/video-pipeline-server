from enum import StrEnum

from pydantic import BaseModel, Field

from src.schema.lesson.lesson_schema import GradeBand, VideoType


class ShotKind(StrEnum):
    """From the lesson tag of the shot's lines. sung: the host sings to camera (lip-sync); number: our
    code draws number_text on screen; music: no lyrics (intro, outro); unclear: intake doubts the words."""

    SUNG = "sung"
    NUMBER = "number"
    ILLUSTRATE = "illustrate"
    UNCLEAR = "unclear"
    MUSIC = "music"


class CastKind(StrEnum):
    """Only a person is a character; a thing looks and behaves as it really does."""

    PERSON = "person"
    THING = "thing"


class Scale(StrEnum):
    """Ordered largest to smallest; a concept video steps through it one rung per cut."""

    COSMIC = "cosmic"
    PLANET = "planet"
    LANDSCAPE = "landscape"
    HUMAN = "human"
    OBJECT = "object"
    MICROSCOPIC = "microscopic"
    MOLECULAR = "molecular"


class ShotSize(StrEnum):
    """Ordered widest to closest, for the size-jump check."""

    WIDE = "wide"
    MEDIUM_WIDE = "medium_wide"
    MEDIUM = "medium"
    MEDIUM_CLOSE = "medium_close"
    CLOSE = "close"
    EXTREME_CLOSE = "extreme_close"


class Angle(StrEnum):
    EYE = "eye"
    LOW = "low"
    HIGH = "high"
    OVERHEAD = "overhead"


class Position(StrEnum):
    """Where the main subject sits; a number shot keeps a side third open for the figure."""

    LEFT_THIRD = "left_third"
    CENTRE = "centre"
    RIGHT_THIRD = "right_third"


class CameraMove(StrEnum):
    STATIC = "static"
    PAN = "pan"
    TILT = "tilt"
    PUSH_IN = "push_in"
    PULL_BACK = "pull_back"
    TRUCK = "truck"
    PEDESTAL = "pedestal"
    ORBIT = "orbit"
    FOLLOW = "follow"


class Direction(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    TOWARD = "toward"
    AWAY = "away"


class Speed(StrEnum):
    SLOW = "slow"
    MEDIUM = "medium"
    FAST = "fast"


class Motion(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ScreenDirection(StrEnum):
    """Which way the main movement crosses the frame; kept steady within a place (180-degree rule)."""

    LEFT_TO_RIGHT = "left_to_right"
    RIGHT_TO_LEFT = "right_to_left"
    TOWARD = "toward"
    AWAY = "away"
    NONE = "none"


class Transition(StrEnum):
    CUT = "cut"
    MATCH_CUT = "match_cut"
    DISSOLVE = "dissolve"


class Continuity(StrEnum):
    """How a shot links to the one before it; worked out by code from place and subjects."""

    SAME_PLACE = "same_place"
    SAME_SUBJECT = "same_subject"
    NEW_SCENE = "new_scene"


class CastMember(BaseModel):
    """A person or thing that appears in more than one shot. look is copied word for word into every
    image prompt it appears in, next to its reference sheet. scale: the rung it is seen at, so a
    hose-sized cable is never the subject of a planet-wide shot."""

    id: str
    kind: CastKind
    look: str
    size: str
    scale: Scale


class Place(BaseModel):
    """walkable: a person could really stand here; the host sings only from walkable places. host_outfit:
    what the host wears when seen here, fitting the place and the topic; None where the host never appears."""

    id: str
    look: str
    lighting: str
    scale: Scale
    walkable: bool
    host_outfit: str | None = None


class Motif(BaseModel):
    """A visual code kept for the whole video, e.g. "antibiotic" / "always a cyan liquid"."""

    thing: str
    rule: str


class Camera(BaseModel):
    move: CameraMove
    direction: Direction | None = None
    speed: Speed | None = None


class Slot(BaseModel):
    """A shot's place in time, cut by code from the lyric timings and beats. repeat_of: the slot whose
    shot this one copies (a chorus sung again)."""

    id: str
    start: float
    end: float
    lines: list[int]
    kind: ShotKind
    number_text: str | None = None
    repeat_of: str | None = None


class ShotDraft(BaseModel):
    """What the model writes for one slot."""

    slot: str
    place: str
    subjects: list[str]
    size: ShotSize
    angle: Angle
    position: Position
    start_state: str
    action: str
    end_state: str
    camera: Camera
    motion: Motion
    screen_direction: ScreenDirection
    transition_in: Transition


class StoryboardDraft(BaseModel):
    """What the model writes for the whole video. The host is given, not written; each place says what
    the host wears there."""

    shape: str
    cast: list[CastMember]
    places: list[Place]
    motifs: list[Motif]
    shots: list[ShotDraft]


class ShotPatch(BaseModel):
    """A re-ask's answer: replacements for the named shots only."""

    shots: list[ShotDraft]


class Shot(ShotDraft):
    """A checked shot with its timing and what code decided."""

    id: str
    start: float
    end: float
    lines: list[int]
    kind: ShotKind
    number_text: str | None = None
    repeat_of: str | None = None
    performer: str | None = None
    continuity: Continuity


class Storyboard(BaseModel):
    """Stage 3 output. flags: notes for later stages that are not errors (e.g. attach a returning
    thing's reference sheet)."""

    grade_band: GradeBand
    video_type: VideoType
    shape: str
    cast: list[CastMember]
    places: list[Place]
    motifs: list[Motif]
    shots: list[Shot]
    flags: list[str] = Field(default_factory=list)
    model: str
    prompt_version: str
