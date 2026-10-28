from datetime import datetime, timedelta

import numpy as np
import pytest
from dependency_injector import providers

from src.core.container import Container
from src.schema.song.song_schema import Song, SongStatus


class FakeFfmpeg:
    def decode(self, audio, sample_rate):
        return np.zeros(sample_rate * 4, dtype=np.float32)


class FakeAligner:
    version = "fake-align"

    def __init__(self, confidence=0.85):
        self.calls = 0
        self.confidence = confidence

    def align(self, samples, tokens):
        self.calls += 1
        return [(i * 0.5, i * 0.5 + 0.4, self.confidence) for i in range(len(tokens))]


class FakeBeats:
    version = "fake-beats"

    def track(self, samples):
        return [0.5, 1.0, 1.5, 2.0], [0.5, 2.0]


LYRICS = "This is a colony.\nIn 1858 it grew.\n"


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "audio.mp3"
    path.write_bytes(b"not really audio")
    return path


def make(container: Container, aligner: FakeAligner):
    container.ffmpeg_service.override(providers.Object(FakeFfmpeg()))
    container.forced_align_service.override(providers.Object(aligner))
    container.beat_service.override(providers.Object(FakeBeats()))
    return container.intake_service()


def add_song(container, audio, name="colony", lyrics=LYRICS) -> Song:
    return container.song_repository().songs.add(Song(name=name, audio_path=str(audio), lyrics=lyrics))


def test_intake_times_every_word_and_marks_the_song_ready(container, audio):
    service = make(container, FakeAligner())
    song = add_song(container, audio)
    analysis = service.run(song.name)
    assert analysis.accepted and [w.text for w in analysis.words][-3:] == ["1858", "it", "grew."]
    assert [l.index for l in analysis.lines] == [0, 1] and analysis.bpm == 120.0
    stored = container.song_repository().songs.get(song.id)
    assert stored.status == SongStatus.READY and stored.analysis_id == analysis.id
    assert service.analysis_for(song.id).id == analysis.id


def test_the_same_audio_and_words_are_analysed_once(container, audio):
    aligner = FakeAligner()
    service = make(container, aligner)
    first = service.run(add_song(container, audio, "a").id)
    second = service.run(add_song(container, audio, "b").id)
    assert aligner.calls == 1 and second.id == first.id


def test_changed_words_are_analysed_again(container, audio):
    aligner = FakeAligner()
    service = make(container, aligner)
    service.run(add_song(container, audio, "a").id)
    service.run(add_song(container, audio, "b", lyrics=LYRICS + "It grew more.").id)
    assert aligner.calls == 2


def test_force_analyses_again_in_place(container, audio):
    aligner = FakeAligner()
    service = make(container, aligner)
    song = add_song(container, audio)
    first = service.run(song.id)
    again = service.run(song.id, force=True)
    assert aligner.calls == 2 and again.id == first.id


def test_audio_that_does_not_match_the_lyrics_is_rejected(container, audio):
    service = make(container, FakeAligner(confidence=0.2))
    song = add_song(container, audio)
    analysis = service.run(song.id)
    assert not analysis.accepted and "right audio" in analysis.reason
    assert all(l.unclear for l in analysis.lines)
    stored = container.song_repository().songs.get(song.id)
    assert stored.status == SongStatus.REJECTED and stored.reason == analysis.reason


def test_a_crash_marks_the_song_failed(container, audio):
    class Broken(FakeAligner):
        def align(self, samples, tokens):
            raise RuntimeError("model missing")

    service = make(container, Broken())
    song = add_song(container, audio)
    with pytest.raises(RuntimeError):
        service.run(song.id)
    stored = container.song_repository().songs.get(song.id)
    assert stored.status == SongStatus.FAILED and stored.reason == "model missing"


def test_unanalysed_or_missing_songs_are_lookup_errors(container, audio):
    service = make(container, FakeAligner())
    with pytest.raises(LookupError):
        service.run("nope")
    with pytest.raises(LookupError, match="not been analysed"):
        service.analysis_for(add_song(container, audio).id)


def test_a_job_song_is_analysed_without_a_song_row(container, audio):
    service = make(container, FakeAligner())
    expires = datetime(2026, 11, 4)
    analysis = service.analyse(audio, "[verse]\n" + LYRICS, expires_at=expires)
    assert analysis.accepted and len(analysis.lines) == 2 and analysis.expires_at == expires
    assert container.song_repository().songs.list() == []


def test_a_library_song_keeps_its_analysis_forever(container, audio):
    service = make(container, FakeAligner())
    assert service.run(add_song(container, audio).id).expires_at is None


def test_a_shared_analysis_keeps_the_longest_retention(container, audio):
    aligner = FakeAligner()
    service = make(container, aligner)
    soon, later = datetime(2026, 11, 4), datetime(2026, 11, 4) + timedelta(days=3)
    assert service.analyse(audio, LYRICS, expires_at=later).expires_at == later
    assert service.analyse(audio, LYRICS, expires_at=soon).expires_at == later
    assert service.run(add_song(container, audio).id).expires_at is None
    assert service.analyse(audio, LYRICS, expires_at=soon).expires_at is None
    assert aligner.calls == 1
