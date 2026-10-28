import shutil
from pathlib import Path

from src.core.configs import Settings
from src.repository.song.song_repository import SongRepository
from src.schema.song.song_schema import Song
from src.service.intake.intake_lyrics import clean_lyrics


class SongService:
    """Registers library songs. The audio is copied under the asset folder, so a song keeps working when the
    file it came from is moved or deleted."""

    def __init__(self, song_repository: SongRepository, settings: Settings) -> None:
        self._repo = song_repository
        self._settings = settings

    def add(self, name: str, audio: Path, lyrics: str) -> Song:
        if self._repo.songs.find_one(name=name) is not None:
            raise ValueError(f"A song named {name!r} already exists")
        if not Path(audio).is_file():
            raise ValueError(f"Audio file {audio} not found")
        if not clean_lyrics(lyrics):
            raise ValueError("The lyrics have no lines to align")
        song = Song(name=name, audio_path="", lyrics=lyrics)
        target = Path(self._settings.asset_dir) / "songs" / song.id / f"audio{Path(audio).suffix.lower()}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(audio, target)
        song.audio_path = str(target.resolve())
        return self._repo.songs.add(song)

    def get(self, ref: str) -> Song:
        song = self._repo.find_song(ref)
        if song is None:
            raise LookupError(f"Song {ref} not found")
        return song

    def list(self) -> list[Song]:
        return self._repo.songs.list()
