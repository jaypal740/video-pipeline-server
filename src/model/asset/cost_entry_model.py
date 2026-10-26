from sqlalchemy import Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from src.model.base_model import Base, EntityMixin


class CostEntryRow(EntityMixin, Base):
    __tablename__ = "cost_entries"

    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), nullable=False, index=True)
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False)
    target_id: Mapped[str | None] = mapped_column(String(64))
    asset_id: Mapped[str | None] = mapped_column(String(32))
    units: Mapped[float | None] = mapped_column(Float)
    unit: Mapped[str | None] = mapped_column(String(32))
