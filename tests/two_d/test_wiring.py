import json
from pathlib import Path

import pytest

from src.core.container import Container
from src.schema.enums import StageName, StageStatus
from src.schema.lesson.lesson_schema import GradeBand
from src.schema.run.run_schema import Run
from src.schema.run.stage_state_schema import StageState
from src.schema.song.song_schema import Song, SongStatus
from src.schema.two_d.script_schema import AnimationScript
from src.service.two_d.drawing.drawing_service import DrawingService
from src.service.two_d.lesson.beat_lesson_service import BeatLessonService
from src.service.two_d.loop.loop_service import LoopService
from src.service.two_d.render.film_check_service import FilmCheckService
from src.service.two_d.render.render_service import RenderService
from src.service.two_d.script.script_service import ScriptService

STORED_SCRIPT = Path(__file__).parent / "fixtures" / "water_cycle_script.json"


def test_container_has_2d_services(container: Container):
    assert isinstance(container.beat_lesson_service(), BeatLessonService)
    assert isinstance(container.script_service(), ScriptService)
    assert isinstance(container.drawing_service(), DrawingService)
    assert isinstance(container.loop_service(), LoopService)
    assert isinstance(container.render_service(), RenderService)
    assert isinstance(container.film_check_service(), FilmCheckService)


def test_stored_2d_script_still_parses():
    script = AnimationScript(**json.loads(STORED_SCRIPT.read_text()))
    assert len(script.beats) == 2 and script.style.startswith("A bold 2D cartoon")


@pytest.mark.parametrize("service", ["drawing_service", "loop_service", "render_service"])
def test_2d_stages_refuse_a_narration_run(container: Container, service: str):
    runs = container.run_repository()
    song = container.song_repository().songs.add(Song(name="s", audio_path="a.mp3", lyrics="x", status=SongStatus.READY))
    run = runs.runs.add(Run(song_id=song.id, name="s", audience=GradeBand.G3_5))
    runs.stage_states.add(StageState(run_id=run.id, stage=StageName.LESSON, status=StageStatus.DONE,
                                     output={"lines": [], "film_type": "narration"}))
    with pytest.raises(ValueError, match="is a narration film; use `run <stage>`"):
        getattr(container, service)().run(run.id)
