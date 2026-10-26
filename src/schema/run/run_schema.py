from pydantic import Field

from src.schema.base_schema import Entity
from src.schema.enums import RunStatus, StageName


class Run(Entity):
    """One video being made: its inputs, where it stands, and the model versions it is pinned to."""

    name: str
    audio_path: str
    script_path: str
    audience: str | None = None
    brief: str | None = None
    status: RunStatus = RunStatus.CREATED
    current_stage: StageName = StageName.INTAKE
    model_pins: dict = Field(default_factory=dict)
