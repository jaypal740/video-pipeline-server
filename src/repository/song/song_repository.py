from sqlalchemy.orm import Session, sessionmaker

from src.model.song.song_analysis_model import SongAnalysisRow
from src.model.song.song_model import SongRow
from src.repository.base_repository import BaseRepository
from src.schema.song.song_analysis_schema import SongAnalysis
from src.schema.song.song_schema import Song


class SongRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory
        self.songs = BaseRepository(session_factory, SongRow, Song)
        self.analyses = BaseRepository(session_factory, SongAnalysisRow, SongAnalysis)

    def find_song(self, ref: str, session: Session | None = None) -> Song | None:
        """A song by its id or its name."""
        return self.songs.get(ref, session=session) or self.songs.find_one(session=session, name=ref)

    def find_analysis(self, cache_key: str, session: Session | None = None) -> SongAnalysis | None:
        return self.analyses.find_one(session=session, cache_key=cache_key)
