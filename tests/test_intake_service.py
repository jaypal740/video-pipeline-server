import numpy as np
import pytest
from dependency_injector import providers

from src.core.container import Container
from src.schema.enums import StageName, StageStatus
from src.schema.run.run_schema import Run


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


@pytest.fixture
def song(tmp_path):
    audio, lyrics = tmp_path / "audio.mp3", tmp_path / "lyrics.txt"
    audio.write_bytes(b"not really audio")
    lyrics.write_text("This is a colony.\nIn 1858 it grew.\n")
    return audio, lyrics


def make(container: Container, aligner: FakeAligner):
    container.ffmpeg_service.override(providers.Object(FakeFfmpeg()))
    container.forced_align_service.override(providers.Object(aligner))
    container.beat_service.override(providers.Object(FakeBeats()))
    return container.intake_service()


def add_run(container, song):
    audio, lyrics = song
    return container.run_repository().runs.add(Run(name="t", audio_path=str(audio), script_path=str(lyrics)))


def test_intake_times_every_word_and_marks_the_stage_done(container, song):
    service = make(container, FakeAligner())
    run = add_run(container, song)
    analysis = service.run(run.id)
    assert analysis.accepted and [w.text for w in analysis.words][-3:] == ["1858", "it", "grew."]
    assert [l.index for l in analysis.lines] == [0, 1] and analysis.bpm == 120.0
    state = container.run_repository().stage(run.id, StageName.INTAKE)
    assert state.status == StageStatus.DONE and state.output == {"analysis_id": analysis.id, "cache_hit": False}
    assert service.analysis_for(run.id).id == analysis.id


def test_the_same_song_is_analysed_once(container, song):
    aligner = FakeAligner()
    service = make(container, aligner)
    first = service.run(add_run(container, song).id)
    second_run = add_run(container, song)
    second = service.run(second_run.id)
    assert aligner.calls == 1 and second.id == first.id
    assert container.run_repository().stage(second_run.id, StageName.INTAKE).output["cache_hit"] is True


def test_force_analyses_again_in_place(container, song):
    aligner = FakeAligner()
    service = make(container, aligner)
    run = add_run(container, song)
    first = service.run(run.id)
    again = service.run(run.id, force=True)
    assert aligner.calls == 2 and again.id == first.id
    assert container.run_repository().stage(run.id, StageName.INTAKE).attempts == 2


def test_audio_that_does_not_match_the_lyrics_is_rejected(container, song):
    service = make(container, FakeAligner(confidence=0.2))
    run = add_run(container, song)
    analysis = service.run(run.id)
    assert not analysis.accepted and "right audio" in analysis.reason
    assert all(l.unclear for l in analysis.lines)
    state = container.run_repository().stage(run.id, StageName.INTAKE)
    assert state.status == StageStatus.FAILED and state.error == analysis.reason


def test_a_crash_marks_the_stage_failed(container, song):
    class Broken(FakeAligner):
        def align(self, samples, tokens):
            raise RuntimeError("model missing")

    service = make(container, Broken())
    run = add_run(container, song)
    with pytest.raises(RuntimeError):
        service.run(run.id)
    state = container.run_repository().stage(run.id, StageName.INTAKE)
    assert state.status == StageStatus.FAILED and state.error == "model missing"


def test_missing_run_is_a_lookup_error(container):
    with pytest.raises(LookupError):
        make(container, FakeAligner()).run("nope")
