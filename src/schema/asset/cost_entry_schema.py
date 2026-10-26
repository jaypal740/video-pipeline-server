from src.schema.base_schema import Entity
from src.schema.enums import StageName


class CostEntry(Entity):
    """One paid call. Summed per shot, per stage and per run."""

    run_id: str
    stage: StageName
    provider: str
    model: str
    cost_usd: float
    target_id: str | None = None
    asset_id: str | None = None
    units: float | None = None
    unit: str | None = None
