import pytest

from src.core.container import Container
from src.schema.song.song_schema import SongStatus


@pytest.fixture
def service(container: Container, tmp_path):
    container.settings().asset_dir = str(tmp_path / "assets")
    return container.song_service()


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "Song.MP3"
    path.write_bytes(b"audio bytes")
    return path


def test_add_copies_the_audio_and_keeps_the_lyrics(service, audio):
    song = service.add("ocean-salty", audio, "Why is the ocean salty?")
    assert song.status == SongStatus.PENDING and song.lyrics == "Why is the ocean salty?"
    assert song.audio_path.endswith(f"songs/{song.id}/audio.mp3")
    audio.unlink()                                        # the original can go; the song keeps its copy
    assert service.get("ocean-salty").audio_path == song.audio_path
    assert open(song.audio_path, "rb").read() == b"audio bytes"


def test_names_are_unique(service, audio):
    service.add("ocean-salty", audio, "Salt.")
    with pytest.raises(ValueError, match="already exists"):
        service.add("ocean-salty", audio, "Salt.")


def test_missing_audio_or_empty_lyrics_are_refused(service, audio, tmp_path):
    with pytest.raises(ValueError, match="not found"):
        service.add("a", tmp_path / "nope.mp3", "Salt.")
    with pytest.raises(ValueError, match="no lines"):
        service.add("b", audio, "# only a comment\n\n")


def test_unknown_song_is_a_lookup_error(service):
    with pytest.raises(LookupError):
        service.get("missing")
