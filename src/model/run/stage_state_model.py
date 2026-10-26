from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from src.model.base_model import Base, EntityMixin


class StageStateRow(EntityMixin, Base):
    __tablename__ = "stage_states"
    __table_args__ = (UniqueConstraint("run_id", "stage"),)

    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), nullable=False, index=True)
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False)
    input_hash: Mapped[str | None] = mapped_column(String(64))
    output: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
