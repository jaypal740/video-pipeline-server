from dependency_injector import providers

from src.core.container import Container
from src.repository.asset.asset_repository import AssetRepository
from src.repository.run.run_repository import RunRepository


def test_builds_repos(container: Container):
    assert isinstance(container.run_repository(), RunRepository)
    assert isinstance(container.asset_repository(), AssetRepository)


def test_singletons(container: Container):
    assert container.engine() is container.engine()
    assert container.run_repository() is container.run_repository()


def test_override(container: Container):
    fake = object()
    with container.run_repository.override(providers.Object(fake)):
        assert container.run_repository() is fake
    assert container.run_repository() is not fake


def test_uses_overridden_settings(container: Container):
    assert container.engine().url.database == ":memory:"
