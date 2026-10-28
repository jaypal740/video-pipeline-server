from dependency_injector import containers, providers

from src.core.configs import Settings
from src.core.database import make_engine, make_session_factory
from src.core.unit_of_work import UnitOfWork
from src.repository.asset.asset_repository import AssetRepository
from src.repository.run.run_repository import RunRepository
from src.repository.song.song_repository import SongRepository
from src.service.intake.intake_service import IntakeService
from src.service.intake.preview_service import PreviewService
from src.service.local.beat_service import BeatService
from src.service.local.ffmpeg_service import FfmpegService
from src.service.local.forced_align_service import ForcedAlignService
from src.service.song.song_service import SongService


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
    song_repository = providers.Singleton(SongRepository, session_factory=session_factory)

    # local tools
    ffmpeg_service = providers.Singleton(FfmpegService)
    forced_align_service = providers.Singleton(ForcedAlignService, device=settings.provided.device)
    beat_service = providers.Singleton(BeatService, device=settings.provided.device)

    # services
    song_service = providers.Singleton(SongService, song_repository=song_repository, settings=settings)
    intake_service = providers.Singleton(
        IntakeService,
        song_repository=song_repository,
        ffmpeg=ffmpeg_service,
        aligner=forced_align_service,
        beats=beat_service,
    )
    preview_service = providers.Singleton(PreviewService, ffmpeg=ffmpeg_service)
