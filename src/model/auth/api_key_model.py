from datetime import datetime

from sqlalchemy import DateTime, Float, String
from sqlalchemy.orm import Mapped, mapped_column

from src.model.base_model import Base, EntityMixin


class ApiKeyRow(EntityMixin, Base):
    __tablename__ = "api_keys"

    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    prefix: Mapped[str] = mapped_column(String(32), nullable=False, unique=True)
    key_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    monthly_cap_usd: Mapped[float | None] = mapped_column(Float)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime)
