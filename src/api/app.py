from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.router import health_router, song_router
from src.core.container import Container
from src.core.database import create_tables
from src.core.logging import configure_logging
from src.service.auth.auth_service import InvalidApiKey, RevokedApiKey


def create_app(container: Container) -> FastAPI:
    """The pipeline server products call with their API key. Every route resolves services from the
    container; nothing touches the DB directly."""

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        settings = container.settings()
        configure_logging(settings.log_level, settings.log_dir)
        create_tables(container.engine())
        yield

    app = FastAPI(title="Video Pipeline", lifespan=lifespan)
    app.state.container = container
    for module in (health_router, song_router):
        app.include_router(module.router)

    @app.exception_handler(InvalidApiKey)
    async def _unauthorised(_: Request, exc: InvalidApiKey) -> JSONResponse:
        return JSONResponse(status_code=401, content={"detail": str(exc)}, headers={"WWW-Authenticate": "Bearer"})

    @app.exception_handler(RevokedApiKey)
    async def _revoked(_: Request, exc: RevokedApiKey) -> JSONResponse:
        return JSONResponse(status_code=403, content={"detail": str(exc)})

    @app.exception_handler(LookupError)
    async def _not_found(_: Request, exc: LookupError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc).strip("'\"")})

    @app.exception_handler(ValueError)
    async def _invalid(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    return app


app = create_app(Container())
