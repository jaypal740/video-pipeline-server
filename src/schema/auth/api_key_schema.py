from datetime import datetime

from src.schema.base_schema import Entity


class ApiKey(Entity):
    """One product's key to the pipeline server. Only the SHA-256 of the key is kept; the key itself is
    shown once, when it is created. prefix is its first characters, used to find it and to name it."""

    name: str
    prefix: str
    key_hash: str
    monthly_cap_usd: float | None = None
    revoked_at: datetime | None = None
    last_used_at: datetime | None = None
