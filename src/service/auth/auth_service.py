import hashlib
import hmac
import secrets

from src.repository.auth.auth_repository import AuthRepository
from src.schema.auth.api_key_schema import ApiKey
from src.schema.base_schema import utcnow

# vp_<12 hex>_<secret>: the first 15 characters are the prefix stored in clear to find the key.
KEY_START = "vp_"
PREFIX_LENGTH = len(KEY_START) + 12


class InvalidApiKey(Exception):
    """No key, or a key the server never issued."""


class RevokedApiKey(Exception):
    """A real key that has been revoked."""


def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


class AuthService:
    """Issues, checks and revokes the API keys products call the server with."""

    def __init__(self, auth_repository: AuthRepository) -> None:
        self._repo = auth_repository

    def create(self, name: str, monthly_cap_usd: float | None = None) -> tuple[ApiKey, str]:
        """A new key for a product. Returns the stored record and the key itself, which is never stored."""
        raw = f"{KEY_START}{secrets.token_hex(6)}_{secrets.token_urlsafe(32)}"
        key = ApiKey(name=name, prefix=raw[:PREFIX_LENGTH], key_hash=hash_key(raw), monthly_cap_usd=monthly_cap_usd)
        return self._repo.api_keys.add(key), raw

    def verify(self, raw: str | None) -> ApiKey:
        """The key a request was made with; records when it was last used."""
        if not raw or not raw.startswith(KEY_START):
            raise InvalidApiKey("Missing or malformed API key")
        key = self._repo.find_key(raw[:PREFIX_LENGTH])
        if key is None or not hmac.compare_digest(key.key_hash, hash_key(raw)):
            raise InvalidApiKey("Invalid API key")
        if key.revoked_at is not None:
            raise RevokedApiKey(f"API key {key.prefix} has been revoked")
        key.last_used_at = utcnow()
        return self._repo.api_keys.update(key)

    def revoke(self, prefix: str) -> ApiKey:
        key = self._repo.find_key(prefix)
        if key is None:
            raise LookupError(f"API key {prefix} not found")
        if key.revoked_at is None:
            key.revoked_at = utcnow()
            key = self._repo.api_keys.update(key)
        return key

    def list(self) -> list[ApiKey]:
        return self._repo.api_keys.list()
