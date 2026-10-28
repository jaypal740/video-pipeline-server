from sqlalchemy import JSON, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.model.base_model import Base, EntityMixin


class RunRow(EntityMixin, Base):
    __tablename__ = "runs"

    song_id: Mapped[str] = mapped_column(ForeignKey("songs.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    audience: Mapped[str | None] = mapped_column(String(64))
    brief: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    current_stage: Mapped[str] = mapped_column(String(32), nullable=False)
    model_pins: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
