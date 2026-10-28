from datetime import datetime

from pydantic import BaseModel, Field

from src.schema.base_schema import Entity


class TimedWord(BaseModel):
    """One word of the lyrics as written. confidence is None when the aligner could not time it
    (punctuation-only tokens); its times then sit between its neighbours."""

    index: int
    line_index: int
    text: str
    start: float
    end: float
    confidence: float | None = None


class LyricLine(BaseModel):
    """One line of the lyrics. unclear: the audio may not say these words, so no shot is built on them
    until a person checks the line."""

    index: int
    text: str
    start: float
    end: float
    confidence: float
    unclear: bool = False


class SongAnalysis(Entity):
    """Stage 1 output for one audio + lyrics pair. cache_key is reused by every run of the same song.
    expires_at: None for library songs (kept forever); a job's song expires with the job's files."""

    cache_key: str
    audio_sha256: str
    lyrics_sha256: str
    duration_s: float
    accepted: bool
    reason: str | None = None
    words: list[TimedWord] = Field(default_factory=list)
    lines: list[LyricLine] = Field(default_factory=list)
    bpm: float | None = None
    beats: list[float] = Field(default_factory=list)
    downbeats: list[float] = Field(default_factory=list)
    beat_strength: float = 0.0
    tool_versions: dict = Field(default_factory=dict)
    expires_at: datetime | None = None
