from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.model.base_model import Base, EntityMixin


class SongRow(EntityMixin, Base):
    __tablename__ = "songs"

    name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    audio_path: Mapped[str] = mapped_column(Text, nullable=False)
    lyrics: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    analysis_id: Mapped[str | None] = mapped_column(ForeignKey("song_analyses.id"))
    reason: Mapped[str | None] = mapped_column(Text)
