from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.model import Base

_MEMORY_URL = "sqlite:///:memory:"
_SQLITE_PREFIX = "sqlite:///"


def make_engine(url: str) -> Engine:
    if url == _MEMORY_URL:
        return create_engine(url, connect_args={"check_same_thread": False}, poolclass=StaticPool)
    if url.startswith(_SQLITE_PREFIX):
        Path(url.removeprefix(_SQLITE_PREFIX)).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


def create_tables(engine: Engine) -> None:
    Base.metadata.create_all(engine)
