from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from src.model.asset.asset_model import AssetRow
from src.model.asset.cost_entry_model import CostEntryRow
from src.repository.base_repository import BaseRepository, session_scope
from src.schema.asset.asset_schema import Asset
from src.schema.asset.cost_entry_schema import CostEntry


class AssetRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory
        self.assets = BaseRepository(session_factory, AssetRow, Asset)
        self.cost_entries = BaseRepository(session_factory, CostEntryRow, CostEntry)

    def find_cached(self, cache_key: str, session: Session | None = None) -> Asset | None:
        return self.assets.find_one(session=session, cache_key=cache_key)

    def run_cost(self, run_id: str, session: Session | None = None) -> float:
        stmt = select(func.coalesce(func.sum(CostEntryRow.cost_usd), 0.0)).where(CostEntryRow.run_id == run_id)
        with session_scope(self._session_factory, session) as s:
            return float(s.scalar(stmt))
