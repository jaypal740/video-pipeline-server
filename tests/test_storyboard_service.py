import pytest
from dependency_injector import providers

from src.core.container import Container
from src.schema.enums import StageName, StageStatus
from src.schema.lesson.lesson_schema import GradeBand, LessonLine, LessonPlan, LineTag, VideoType
from src.schema.run.run_schema import Run
from src.schema.run.stage_state_schema import StageState
from src.schema.song.song_analysis_schema import LyricLine, SongAnalysis
from src.schema.song.song_schema import Song, SongStatus
from src.schema.storyboard.storyboard_schema import Continuity, Position
from src.service.llm.llm_service import ProviderReply

LINES = [LyricLine(index=i, text=t, start=i * 4, end=i * 4 + 3.8, confidence=0.9)
         for i, t in enumerate(["This is a colony.", "In 1858 it grew.", "It fills the dish."])]
TAGS = [LineTag.ILLUSTRATE, LineTag.NUMBER, LineTag.ILLUSTRATE]


def shot(slot, size="medium", action="The colony spreads across the dish"):
    return {"slot": slot, "place": "dish", "subjects": ["colony"], "size": size, "angle": "high",
            "position": "centre", "start_state": "A small teal colony sits in a clear dish", "action": action,
            "end_state": "The colony covers half the dish", "camera": {"move": "push_in", "speed": "slow"},
            "motion": "low", "screen_direction": "none", "transition_in": "cut"}


def board(*shots, cast=None):
    return {"shape": "From one colony to a full dish", "motifs": [],
            "cast": cast or [{"id": "colony", "kind": "thing", "look": "A teal bacterial colony", "size": "a few mm",
                              "scale": "object"}],
            "places": [{"id": "dish", "look": "A clear glass petri dish on a white bench", "lighting": "soft daylight",
                        "scale": "object", "walkable": True, "host_outfit": "science_club"}],
            "shots": list(shots) or [shot("s01", size="wide"), shot("s02"), shot("s03")]}


class FakeClaude:
    name = "anthropic"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def parse(self, model, system, user, output_type, effort):
        self.prompts.append(user)
        return ProviderReply(output=output_type(**self.replies.pop(0)), model=model, input_tokens=1000,
                             output_tokens=500, cost_usd=0.05)


@pytest.fixture
def run(container: Container) -> Run:
    songs, runs = container.song_repository(), container.run_repository()
    analysis = songs.analyses.add(SongAnalysis(cache_key="k", audio_sha256="a", lyrics_sha256="l", duration_s=12,
                                               accepted=True, lines=LINES, beats=[4.0, 8.0]))
    song = songs.songs.add(Song(name="colony", audio_path="a.mp3", lyrics="x", status=SongStatus.READY,
                                analysis_id=analysis.id))
    run = runs.runs.add(Run(song_id=song.id, name="colony", audience=GradeBand.G3_5))
    plan = LessonPlan(topic="Bacteria", subject="Science", summary="S.", video_type=VideoType.PROCESS,
                      type_reason="r", key_terms=[], must_not_teach=[], grade_band=GradeBand.G3_5, model="m",
                      prompt_version="v", lines=[LessonLine(index=i, teaches="t", show="s", tag=tag,
                                                            number_text="1858" if tag == LineTag.NUMBER else None)
                                                 for i, tag in enumerate(TAGS)])
    runs.stage_states.add(StageState(run_id=run.id, stage=StageName.LESSON, status=StageStatus.DONE,
                                     output=plan.model_dump(mode="json")))
    return run


def storyboard(container: Container, claude: FakeClaude):
    container.anthropic_service.override(providers.Object(claude))
    return container.storyboard_service()


def test_a_board_is_cut_written_fixed_stored_and_costed(container, run):
    claude = FakeClaude(board())
    result = storyboard(container, claude).run(run.id)
    assert [s.id for s in result.shots] == ["s01", "s02", "s03"] and result.cast[0].id == "host"
    assert result.shots[1].number_text == "1858" and result.shots[1].position == Position.RIGHT_THIRD
    assert result.shots[1].continuity == Continuity.SAME_PLACE
    assert "s02 | 4.0-8.0s" in claude.prompts[0] and "code draws 1858" in claude.prompts[0]
    state = container.run_repository().stage(run.id, StageName.STORYBOARD)
    assert state.status == StageStatus.DONE and container.asset_repository().run_cost(run.id) == pytest.approx(0.05)


def test_only_the_failing_shot_is_re_asked(container, run):
    claude = FakeClaude(board(shot("s01", size="wide"), shot("s02", action="A label appears on the dish"), shot("s03")),
                        {"shots": [shot("s02")]})
    result = storyboard(container, claude).run(run.id)
    assert result.shots[1].action == "The colony spreads across the dish"
    assert "- s02:" in claude.prompts[1] and "- s01:" not in claude.prompts[1]
    assert container.asset_repository().run_cost(run.id) == pytest.approx(0.10)


def test_a_bad_cast_re_asks_the_whole_board(container, run):
    bad = [{"id": "host", "kind": "person", "look": "A man", "size": "tall", "scale": "human"}]
    claude = FakeClaude(board(shot("s01", size="wide"), shot("s02"), shot("s03"), cast=bad), board())
    storyboard(container, claude).run(run.id)
    assert len(claude.prompts) == 2 and "'host' is given" in claude.prompts[1]


def test_it_gives_up_after_two_re_asks_and_records_the_reasons(container, run):
    bad = shot("s02", action="A label appears on the dish")
    claude = FakeClaude(board(shot("s01", size="wide"), bad, shot("s03")), {"shots": [bad]}, {"shots": [bad]})
    with pytest.raises(ValueError, match="still has problems"):
        storyboard(container, claude).run(run.id)
    state = container.run_repository().stage(run.id, StageName.STORYBOARD)
    assert state.status == StageStatus.FAILED and "label" in state.error
    assert [s["slot"] for s in state.output["draft"]["shots"]] == ["s01", "s02", "s03"]            # kept to inspect
    assert "s02" in state.output["problems"]
    assert container.asset_repository().run_cost(run.id) == pytest.approx(0.15)


def test_the_same_inputs_reuse_the_board_and_a_lesson_is_required(container, run):
    claude = FakeClaude(board(), board())
    service = storyboard(container, claude)
    service.run(run.id)
    service.run(run.id)
    assert len(claude.prompts) == 1
    service.run(run.id, force=True)
    assert len(claude.prompts) == 2
    other = container.run_repository().runs.add(Run(song_id=run.song_id, name="x", audience=GradeBand.G3_5))
    with pytest.raises(ValueError, match="lesson"):
        service.run(other.id)


def test_a_slot_written_twice_is_fixed_by_the_patch(container, run):
    claude = FakeClaude(board(shot("s01", size="wide"), shot("s02"), shot("s02"), shot("s03")), {"shots": [shot("s02")]})
    result = storyboard(container, claude).run(run.id)
    assert [s.id for s in result.shots] == ["s01", "s02", "s03"] and len(claude.prompts) == 2
    assert "more than one shot" in claude.prompts[1]


def test_shots_for_slots_that_do_not_exist_are_dropped(container, run):
    ghost = {**shot("s99"), "place": "nowhere"}
    claude = FakeClaude(board(shot("s01", size="wide"), shot("s02"), shot("s03"), ghost))
    result = storyboard(container, claude).run(run.id)
    assert [s.id for s in result.shots] == ["s01", "s02", "s03"] and len(claude.prompts) == 1


def test_a_lesson_plan_older_than_the_analysis_is_refused(container, run):
    songs = container.song_repository()
    song = songs.songs.get(run.song_id)
    analysis = songs.analyses.get(song.analysis_id)
    analysis.lines = LINES[:2]
    songs.analyses.update(analysis)
    with pytest.raises(ValueError, match="run the lesson stage again"):
        storyboard(container, FakeClaude()).run(run.id)
    assert container.run_repository().stage(run.id, StageName.STORYBOARD) is None


def test_a_sung_shot_happens_in_a_walkable_place_in_the_chosen_kit(container, run):
    runs = container.run_repository()
    state = runs.stage(run.id, StageName.LESSON)
    state.output["lines"][2]["tag"] = "face_to_camera"
    runs.stage_states.update(state)
    claude = FakeClaude(board(shot("s01", size="wide"), shot("s02"), {**shot("s03"), "subjects": ["host"]}))
    result = storyboard(container, claude).run(run.id)
    sung = result.shots[2]
    assert sung.kind == "sung" and sung.place == "dish" and sung.performer == "host"
    assert result.places[0].host_outfit == "science_club" and "Wardrobe kits: everyday:" in claude.prompts[0]
