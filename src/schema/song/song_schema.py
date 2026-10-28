from enum import StrEnum

from src.schema.base_schema import Entity


class SongStatus(StrEnum):
    PENDING = "pending"
    ANALYSING = "analysing"
    READY = "ready"
    REJECTED = "rejected"
    FAILED = "failed"


class Song(Entity):
    """One of our own library songs (defaults and test songs), kept permanently with its audio, lyrics
    and intake analysis. Songs a product sends with a job are never stored here."""

    name: str
    audio_path: str
    lyrics: str
    status: SongStatus = SongStatus.PENDING
    analysis_id: str | None = None
    reason: str | None = None
