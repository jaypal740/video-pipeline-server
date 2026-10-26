from dependency_injector import containers, providers

from src.core.configs import Settings
from src.core.database import make_engine, make_session_factory
from src.core.unit_of_work import UnitOfWork
from src.repository.asset.asset_repository import AssetRepository
from src.repository.run.run_repository import RunRepository


class Container(containers.DeclarativeContainer):
    """DI container. One per process; every dependency is a singleton built from settings."""

    wiring_config = containers.WiringConfiguration(modules=[])

    settings = providers.Singleton(Settings)

    # infra
    engine = providers.Singleton(make_engine, url=settings.provided.db_url)
    session_factory = providers.Singleton(make_session_factory, engine=engine)
    unit_of_work = providers.Factory(UnitOfWork, session_factory=session_factory)

    # repositories
    run_repository = providers.Singleton(RunRepository, session_factory=session_factory)
    asset_repository = providers.Singleton(AssetRepository, session_factory=session_factory)
