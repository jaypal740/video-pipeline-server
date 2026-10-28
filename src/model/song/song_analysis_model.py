from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.model.base_model import Base, EntityMixin


class SongAnalysisRow(EntityMixin, Base):
    __tablename__ = "song_analyses"

    cache_key: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    audio_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    lyrics_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    duration_s: Mapped[float] = mapped_column(Float, nullable=False)
    accepted: Mapped[bool] = mapped_column(Boolean, nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    words: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    lines: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    bpm: Mapped[float | None] = mapped_column(Float)
    beats: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    downbeats: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    beat_strength: Mapped[float] = mapped_column(Float, nullable=False)
    tool_versions: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, index=True)
