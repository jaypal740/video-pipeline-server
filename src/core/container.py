from dependency_injector import containers, providers

from src.core.configs import Settings
from src.core.database import make_engine, make_session_factory
from src.core.unit_of_work import UnitOfWork
from src.repository.asset.asset_repository import AssetRepository
from src.repository.auth.auth_repository import AuthRepository
from src.repository.run.run_repository import RunRepository
from src.repository.song.song_repository import SongRepository
from src.service.auth.auth_service import AuthService
from src.service.intake.intake_service import IntakeService
from src.service.intake.preview_service import PreviewService
from src.service.lesson.lesson_service import LessonService
from src.service.llm.anthropic_service import AnthropicService
from src.service.llm.llm_service import LlmService
from src.service.llm.openai_service import OpenAiService
from src.service.local.beat_service import BeatService
from src.service.local.ffmpeg_service import FfmpegService
from src.service.local.forced_align_service import ForcedAlignService
from src.service.song.song_service import SongService
from src.service.storyboard.storyboard_service import StoryboardService


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
    auth_repository = providers.Singleton(AuthRepository, session_factory=session_factory)

    # local tools
    ffmpeg_service = providers.Singleton(FfmpegService)
    forced_align_service = providers.Singleton(ForcedAlignService, device=settings.provided.device)
    beat_service = providers.Singleton(BeatService, device=settings.provided.device)

    # model providers
    openai_service = providers.Singleton(OpenAiService, api_key=settings.provided.openai_api_key)
    anthropic_service = providers.Singleton(AnthropicService, api_key=settings.provided.claude_api_key)
    llm_service = providers.Singleton(LlmService, openai=openai_service, anthropic=anthropic_service)

    # services
    auth_service = providers.Singleton(AuthService, auth_repository=auth_repository)
    song_service = providers.Singleton(SongService, song_repository=song_repository, settings=settings)
    intake_service = providers.Singleton(
        IntakeService,
        song_repository=song_repository,
        ffmpeg=ffmpeg_service,
        aligner=forced_align_service,
        beats=beat_service,
    )
    preview_service = providers.Singleton(PreviewService, ffmpeg=ffmpeg_service)
    lesson_service = providers.Singleton(
        LessonService,
        run_repository=run_repository,
        song_repository=song_repository,
        asset_repository=asset_repository,
        llm=llm_service,
        settings=settings,
    )
    storyboard_service = providers.Singleton(
        StoryboardService,
        run_repository=run_repository,
        song_repository=song_repository,
        asset_repository=asset_repository,
        llm=llm_service,
        settings=settings,
    )
