from datetime import datetime

from pydantic import Field

from src.schema.base_schema import Entity
from src.schema.enums import StageName, StageStatus


class StageState(Entity):
    """One stage of one run. input_hash makes a re-run skip work whose inputs have not changed."""

    run_id: str
    stage: StageName
    status: StageStatus = StageStatus.PENDING
    attempts: int = 0
    input_hash: str | None = None
    output: dict = Field(default_factory=dict)
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
