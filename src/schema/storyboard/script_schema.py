from enum import StrEnum
from typing import Literal

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
    whose face and mouth show, so it can be lip-synced; name: the stage it stands for (larva), which code
    writes under it in a row or ring."""

    id: str
    kind: DrawingKind
    look: str
    name: str | None = None
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
    stage to the next. ring: the next place round a ring of stages, laid out clockwise from the top, whose
    last stage has an arrow back to the first."""

    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"
    ROW = "row"
    RING = "ring"


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


class ScriptRepair(BaseModel):
    """A small fix to a finished script: the new drawings it needs and the beats that change, each whole."""

    drawings: list[Drawing]
    beats: list[BeatScriptDraft]


class ScriptReview(BaseModel):
    """What a finished script fails to show of what its plan teaches: one sentence each, starting with the
    beat's id. Empty when the script teaches the plan."""

    gaps: list[str]


class FilmProblem(BaseModel):
    """One thing the film shows wrong, from the second `at` on: what a child would see or wrongly believe."""

    at: float
    kind: Literal["wrong_learning", "hidden", "unclear", "stuck"]
    what: str


class FilmReview(BaseModel):
    """What a judge found in frames of the whole film, put together from the still drawings. Empty when the
    film is fit to finish."""

    problems: list[FilmProblem]


class Screen(BaseModel):
    """What is on screen when a beat ends: its backdrop and where each drawing sits. sung: for a hero beat,
    the performer drawings that sang it, each with the seconds it sang from and until."""

    background: str | None = None
    on: dict[str, Pos] = Field(default_factory=dict)
    sung: list[tuple[str, float, float]] = Field(default_factory=list)


class BeatScript(BeatScriptDraft):
    """A checked beat with its timing and type from the lesson, and code's actions added."""

    start: float
    end: float
    type: BeatType


class AnimationScript(BaseModel):
    """Stage 3 output: every drawing the video needs and, beat by beat, what they do. style: the look every
    drawing is made in, from the grade band. gaps: what the review still found missing from the plan."""

    grade_band: GradeBand
    style: str
    palette: list[str]
    performer: str | None = None
    drawings: list[Drawing]
    beats: list[BeatScript]
    gaps: list[str] = Field(default_factory=list)
    model: str
    prompt_version: str
