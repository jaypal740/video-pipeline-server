import pytest
from dependency_injector import providers

from src.core.configs import Settings
from src.core.container import Container
from src.core.database import create_tables


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, db_url="sqlite:///:memory:", log_dir=None)   # tests never write data/logs


@pytest.fixture
def container(settings: Settings) -> Container:
    container = Container()
    container.settings.override(providers.Object(settings))
    create_tables(container.engine())
    yield container
    container.unwire()
