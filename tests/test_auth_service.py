import pytest

from src.core.container import Container
from src.service.auth.auth_service import InvalidApiKey, RevokedApiKey


def test_a_new_key_verifies_and_only_its_hash_is_stored(container: Container):
    service = container.auth_service()
    key, raw = service.create("jamwise", monthly_cap_usd=50)
    stored = container.auth_repository().api_keys.get(key.id)
    assert raw.startswith(key.prefix) and raw not in stored.model_dump_json()
    assert stored.monthly_cap_usd == 50 and stored.last_used_at is None
    assert service.verify(raw).id == key.id
    assert container.auth_repository().api_keys.get(key.id).last_used_at is not None


def test_missing_wrong_and_tampered_keys_are_invalid(container: Container):
    service = container.auth_service()
    _, raw = service.create("jamwise")
    for bad in (None, "", "Bearer x", "vp_000000000000_nope", raw[:-1] + ("A" if raw[-1] != "A" else "B")):
        with pytest.raises(InvalidApiKey):
            service.verify(bad)


def test_a_revoked_key_is_refused(container: Container):
    service = container.auth_service()
    key, raw = service.create("jamwise")
    service.revoke(key.prefix)
    with pytest.raises(RevokedApiKey):
        service.verify(raw)
    with pytest.raises(LookupError):
        service.revoke("vp_missing")
