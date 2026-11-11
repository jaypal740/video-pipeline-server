from enum import StrEnum

from pydantic import BaseModel, Field

from src.schema.lesson.lesson_schema import BeatType, GradeBand


class DrawingKind(StrEnum):
    """performer: whoever sings the song, in one pose, the only drawing with a face; object: a real thing
    the song teaches about; background: a full-frame backdrop; icon: a flat symbol (an arrow)."""

    PERFORMER = "performer"
    OBJECT = "object"
    BACKGROUND = "background"
    ICON = "icon"


class Drawing(BaseModel):
    """One picture, drawn once as a flat cut-out and reused in every beat that names it. look: what to draw;
    of: the drawing this is another pose or state of, so both are drawn as the same thing; motion: how it
    moves in its short loop (a bird's wings flap), None when it holds still; sings: a performer drawing
    whose face and mouth show, so it can be lip-synced."""

    id: str
    kind: DrawingKind
    look: str
    of: str | None = None
    motion: str | None = None
    sings: bool = False


class Verb(StrEnum):
    """Everything that can happen on screen. The model writes the first nine; code adds keyword, number
    and lipsync from the lesson's beats."""

    SHOW = "show"
    HIDE = "hide"
    SWAP = "swap"
    MOVE = "move"
    ZOOM = "zoom"
    HIGHLIGHT = "highlight"
    PUSH_IN = "push_in"
    PULL_BACK = "pull_back"
    PAN = "pan"
    KEYWORD = "keyword"
    NUMBER = "number"
    LIPSYNC = "lipsync"


class Pos(StrEnum):
    """row: the next place in a row of stages, which code lays out left to right with an arrow from each
    stage to the next."""

    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"
    ROW = "row"


class Action(BaseModel):
    """One timed action. at and until are seconds of the song. drawing: what is acted on (none for a camera
    move); to: the drawing a swap brings in; pos: where a show appears, where a move ends, which way a pan
    goes; scale: the size a zoom ends at, 1 being the drawing's normal size; text: the word or figure code
    draws."""

    verb: Verb
    drawing: str | None = None
    to: str | None = None
    pos: Pos | None = None
    scale: float | None = None
    text: str | None = None
    at: float
    until: float | None = None


class DrawingList(BaseModel):
    """What the model writes first. performer: who sings the song and what they look like; palette: the
    video's colours, as hex."""

    performer: str | None = None
    palette: list[str]
    drawings: list[Drawing]


class BeatScriptDraft(BaseModel):
    """One beat as the model writes it: its backdrop and what happens in front of it."""

    beat: str
    background: str
    actions: list[Action]


class ScriptPatch(BaseModel):
    """The scripts for one run of beats."""

    beats: list[BeatScriptDraft]


class Screen(BaseModel):
    """What is on screen when a beat ends: its backdrop and where each drawing sits. singer: the performer
    drawing that sang the beat, for a hero beat."""

    background: str | None = None
    on: dict[str, Pos] = Field(default_factory=dict)
    singer: str | None = None


class BeatScript(BeatScriptDraft):
    """A checked beat with its timing and type from the lesson, and code's actions added."""

    start: float
    end: float
    type: BeatType


class AnimationScript(BaseModel):
    """Stage 3 output: every drawing the video needs and, beat by beat, what they do. style: the look every
    drawing is made in, from the grade band."""

    grade_band: GradeBand
    style: str
    palette: list[str]
    performer: str | None = None
    drawings: list[Drawing]
    beats: list[BeatScript]
    model: str
    prompt_version: str
