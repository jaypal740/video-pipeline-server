from pydantic import Field

from src.schema.base_schema import Entity
from src.schema.enums import AssetKind, StageName


class Asset(Entity):
    """A file a stage produced, with everything needed to reproduce it. cache_key lets a repeat
    call with the same model, prompt, refs, seed and params reuse this file instead of paying again."""

    run_id: str
    stage: StageName
    kind: AssetKind
    path: str
    sha256: str
    target_id: str | None = None
    cache_key: str | None = None
    provider: str | None = None
    model: str | None = None
    model_version: str | None = None
    seed: int | None = None
    prompt: str | None = None
    params: dict = Field(default_factory=dict)
    cost_usd: float | None = None
