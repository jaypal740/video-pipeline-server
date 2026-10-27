from sqlalchemy.orm import Session, sessionmaker

from src.model.song.song_analysis_model import SongAnalysisRow
from src.repository.base_repository import BaseRepository
from src.schema.song.song_analysis_schema import SongAnalysis


class SongRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory
        self.analyses = BaseRepository(session_factory, SongAnalysisRow, SongAnalysis)

    def find_analysis(self, cache_key: str, session: Session | None = None) -> SongAnalysis | None:
        return self.analyses.find_one(session=session, cache_key=cache_key)
