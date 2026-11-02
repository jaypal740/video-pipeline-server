from enum import StrEnum

from pydantic import Field, field_validator

from src.schema.base_schema import Entity
from src.schema.enums import AssetKind, StageName


class ReviewState(StrEnum):
    """Whether an image may be used. Only passed or approved images are ever attached as references; a
    keyframe may also be used_best: no try passed, so the best-scoring one above the floor is used and
    the stage says so. needs_review waits for a person during the build of a stage and blocks what
    depends on it; the server itself never waits."""

    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"
    APPROVED = "approved"
    USED_BEST = "used_best"


class Asset(Entity):
    """A file a stage produced, with everything needed to reproduce it. cache_key lets a repeat
    call with the same model, prompt, refs, seed and params reuse this file instead of paying again.
    run_id is None for assets that belong to a band rather than a run (the hosts)."""

    run_id: str | None
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
    review: ReviewState = ReviewState.PENDING
    review_notes: str | None = None

    @field_validator("review", mode="before")
    @classmethod
    def _pending_when_missing(cls, value):
        return value or ReviewState.PENDING                     # rows written before reviews existed
