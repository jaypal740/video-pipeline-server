from pydantic import Field

from src.schema.base_schema import Entity
from src.schema.enums import RunStatus, StageName


class Run(Entity):
    """One video made from an analysed song: where it stands and the model versions it is pinned to."""

    song_id: str
    name: str
    audience: str | None = None
    brief: str | None = None
    status: RunStatus = RunStatus.CREATED
    current_stage: StageName = StageName.LESSON
    model_pins: dict = Field(default_factory=dict)
