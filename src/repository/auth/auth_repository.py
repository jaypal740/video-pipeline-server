from sqlalchemy.orm import Session, sessionmaker

from src.model.auth.api_key_model import ApiKeyRow
from src.repository.base_repository import BaseRepository
from src.schema.auth.api_key_schema import ApiKey


class AuthRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory
        self.api_keys = BaseRepository(session_factory, ApiKeyRow, ApiKey)

    def find_key(self, prefix: str, session: Session | None = None) -> ApiKey | None:
        return self.api_keys.find_one(session=session, prefix=prefix)
