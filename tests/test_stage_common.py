import pytest

from src.core.container import Container
from src.schema.enums import FilmType, StageName, StageStatus
from src.schema.lesson.lesson_schema import GradeBand
from src.schema.run.run_schema import Run
from src.schema.run.stage_state_schema import StageState
from src.schema.song.song_schema import Song, SongStatus
from src.service.stage.stage_common import REALISTIC, film_type_of, require_film_type, storyboard_of


@pytest.fixture
def runs(container: Container):
    return container.run_repository()


@pytest.fixture
def run(container: Container, runs) -> Run:
    song = container.song_repository().songs.add(Song(name="s", audio_path="a.mp3", lyrics="x", status=SongStatus.READY))
    return runs.runs.add(Run(song_id=song.id, name="s", audience=GradeBand.G3_5))


def store_lesson(runs, run: Run, output: dict) -> None:
    runs.stage_states.add(StageState(run_id=run.id, stage=StageName.LESSON, status=StageStatus.DONE, output=output))


def test_film_type_of_2d_lesson(runs, run):
    store_lesson(runs, run, {"beats": []})
    assert film_type_of(runs, run.id) == FilmType.TWO_D


def test_film_type_of_typed_realistic_lesson(runs, run):
    store_lesson(runs, run, {"lines": [], "film_type": "narration"})
    assert film_type_of(runs, run.id) == FilmType.NARRATION


def test_film_type_of_old_realistic_lesson_is_none(runs, run):
    store_lesson(runs, run, {"lines": []})
    assert film_type_of(runs, run.id) is None
    require_film_type(runs, run.id, REALISTIC)
    with pytest.raises(ValueError, match="use `run <stage>`"):
        require_film_type(runs, run.id, frozenset({FilmType.TWO_D}))


def test_film_type_of_new_run_is_none(runs, run):
    assert film_type_of(runs, run.id) is None


def test_new_run_accepted_by_either(runs, run):
    require_film_type(runs, run.id, REALISTIC)
    require_film_type(runs, run.id, frozenset({FilmType.TWO_D}))


def test_2d_run_refused_by_realistic_stages(runs, run):
    store_lesson(runs, run, {"beats": []})
    with pytest.raises(ValueError, match=r"is a 2d film; use `run 2d <stage>`"):
        require_film_type(runs, run.id, REALISTIC)


def test_storyboard_of_refuses_2d_run(runs, run):
    store_lesson(runs, run, {"beats": []})
    runs.stage_states.add(StageState(run_id=run.id, stage=StageName.STORYBOARD, status=StageStatus.DONE,
                                     output={"drawings": []}))
    with pytest.raises(ValueError, match="is a 2d film"):
        storyboard_of(runs, run.id)
