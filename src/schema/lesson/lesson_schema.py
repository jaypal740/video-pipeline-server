from enum import StrEnum

from pydantic import BaseModel, Field


class GradeBand(StrEnum):
    K_2 = "k-2"
    G3_5 = "3-5"
    G6_8 = "6-8"


class VideoType(StrEnum):
    CONCEPT = "concept"
    PROCESS = "process"
    JOURNEY = "journey"
    PERSONALISED_HERO = "personalised_hero"


class LineTag(StrEnum):
    """What later stages do with a line. face_to_camera: a character sings it to the viewer (lip-sync);
    number: a figure our code draws on screen as it is sung; illustrate: a picture of the idea;
    unclear: intake doubts the audio says these words, so nothing is built on it until a person checks."""

    FACE_TO_CAMERA = "face_to_camera"
    NUMBER = "number"
    ILLUSTRATE = "illustrate"
    UNCLEAR = "unclear"


class BandRules(BaseModel):
    """One grade band's rules, from src/templates/band/<band>.toml."""

    band: GradeBand
    grades: list[int]
    allowed_types: list[VideoType]
    override_types: list[VideoType] = Field(default_factory=list)
    look: str
    shot_seconds: list[float]
    face_to_camera_share: float
    on_screen_text: str
    avoid: list[str]


class FormatRules(BaseModel):
    """One video type, from src/templates/format/<type>.toml. face_to_camera_share is enforced by code."""

    type: VideoType
    pick_when: str
    world: str
    on_camera: str
    face_to_camera_share: float


class KeyTerm(BaseModel):
    term: str
    definition: str


class LessonLine(BaseModel):
    index: int
    teaches: str
    show: str
    tag: LineTag
    number_text: str | None = None


class LessonDraft(BaseModel):
    """What the model writes."""

    topic: str
    subject: str
    summary: str
    video_type: VideoType
    type_reason: str
    key_terms: list[KeyTerm]
    must_not_teach: list[str]
    lines: list[LessonLine]


class LessonPlan(LessonDraft):
    """Stage 2 output: the draft once code has checked it, with what it was made from."""

    grade_band: GradeBand
    model: str
    prompt_version: str
