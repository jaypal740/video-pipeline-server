from enum import StrEnum

from pydantic import BaseModel, Field

from src.schema.lesson.lesson_schema import GradeBand, KeyTerm


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


class BeatLessonDraft(BaseModel):
    """What the model writes for a 2d film: teaching beats instead of tagged lines."""

    topic: str
    subject: str
    summary: str
    key_terms: list[KeyTerm]
    must_not_teach: list[str]
    beats: list[BeatDraft]


class BeatLessonPlan(BeatLessonDraft):
    """Stage 2 output: the draft once code has checked it and set its beats in time, with what it was made
    from."""

    beats: list[Beat]
    grade_band: GradeBand
    model: str
    prompt_version: str
