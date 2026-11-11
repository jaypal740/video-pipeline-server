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
    max_places: int
    max_size_jump: int
    camera_moves: list[str]
    max_people_in_frame: int
    max_camera_speed: str
    # Image prompts: style_name names the medium ("A bold 2D cartoon"), style lists its design features
    # (outlines, shading, proportions). look stays the plain-words version the planners read.
    style: str
    style_name: str


class FormatRules(BaseModel):
    """One video type, from src/templates/format/<type>.toml. face_to_camera_share is enforced by code.
    shape, keep_same and camera guide the storyboard; a scale change of more than max_scale_step between
    shots must be bridged by a dissolve, a match cut or a zoom."""

    type: VideoType
    pick_when: str
    world: str
    on_camera: str
    face_to_camera_share: float
    shape: str
    keep_same: str
    camera: str
    max_scale_step: int | None = None
    # The host's wardrobe kit that usually fits this type; the storyboard may pick another for the topic.
    host_outfit: str


class KeyTerm(BaseModel):
    term: str
    definition: str


class LessonLine(BaseModel):
    index: int
    teaches: str
    show: str
    tag: LineTag
    number_text: str | None = None


class LessonSequence(BaseModel):
    """A run of lines on one topic, shown in one place. change: what visibly changes from its first shot
    to its last; terms: the key terms its lines name and its pictures show; fact: what a viewer must
    understand from the pictures alone, with the sound off; steps: the visible stages that show it, in order;
    shown_terms: key terms its pictures show and name although its lines don't sing them (the corona, when
    explaining the Sun's layers)."""

    first_line: int
    last_line: int
    topic: str
    place: str
    change: str
    terms: list[str] = Field(default_factory=list)
    fact: str = ""
    steps: list[str] = Field(default_factory=list)
    shown_terms: list[str] = Field(default_factory=list)


class BeatType(StrEnum):
    """What a teaching beat is. explain: a new idea; example: an instance of an idea already explained;
    recap: an idea shown again (a chorus coming back); hero: the performer sings the lines to the viewer
    (lip-sync); music: no lyrics (intro, break, outro), added by code."""

    EXPLAIN = "explain"
    EXAMPLE = "example"
    RECAP = "recap"
    HERO = "hero"
    MUSIC = "music"


class BeatDraft(BaseModel):
    """One teaching beat as the model writes it: a run of lines that teach one idea. keyword: the word code
    shows on screen as it is sung; number_text: a figure code draws; avoid: the wrong idea its picture must
    not suggest; hint: what to draw, in plain words."""

    first_line: int
    last_line: int
    type: BeatType
    concept: str
    keyword: str | None = None
    number_text: str | None = None
    avoid: str = ""
    hint: str


class Beat(BaseModel):
    """A checked beat with its timing from the lyrics. lines is empty for a music beat; keyword_at: the
    moment the key word is sung."""

    id: str
    start: float
    end: float
    type: BeatType
    lines: list[int] = Field(default_factory=list)
    concept: str = ""
    keyword: str | None = None
    keyword_at: float | None = None
    number_text: str | None = None
    avoid: str = ""
    hint: str = ""


class LessonDraft(BaseModel):
    """What the model writes."""

    topic: str
    subject: str
    summary: str
    key_terms: list[KeyTerm]
    must_not_teach: list[str]
    beats: list[BeatDraft]


class LessonPlan(LessonDraft):
    """Stage 2 output: the draft once code has checked it and set its beats in time, with what it was made
    from."""

    beats: list[Beat]
    grade_band: GradeBand
    model: str
    prompt_version: str
