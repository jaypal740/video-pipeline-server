import json

import pytest
from dependency_injector import providers

from src.core.container import Container
from src.schema.enums import FilmType, StageName, StageStatus
from src.schema.lesson.lesson_schema import GradeBand, LineTag, VideoType
from src.schema.run.run_schema import Run
from src.schema.run.stage_state_schema import StageState
from src.schema.song.song_analysis_schema import LyricLine, SongAnalysis
from src.schema.song.song_schema import Song, SongStatus
from src.service.lesson.lesson_prompt import JAMWISE_SYSTEM, SYSTEM
from src.service.llm.llm_service import LlmRefusal, ProviderReply

LINES = [LyricLine(index=i, text=t, start=i, end=i + 1, confidence=0.9, unclear=(i == 2))
         for i, t in enumerate(["This is a colony.", "In 1858 it grew.", "Mumble words."])]


def draft(video_type="concept", tags=("illustrate", "number", "illustrate")):
    return {"topic": "Bacteria", "subject": "Science", "summary": "S.", "video_type": video_type,
            "type_reason": "r", "key_terms": [{"term": "colony", "definition": "a group"}], "must_not_teach": [],
            "lines": [{"index": i, "teaches": "t", "show": "s", "tag": tag,
                       "number_text": "1858" if tag == "number" else None} for i, tag in enumerate(tags)],
            "sequences": [{"first_line": 0, "last_line": len(tags) - 1, "topic": "t", "place": "p", "change": "c",
                           "fact": "A colony grows.", "steps": ["A colony spreads"]}]}


class FakeClaude:
    name = "anthropic"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0
        self.systems = []

    def parse(self, model, system, user, output_type, effort):
        self.calls += 1
        self.systems.append(system)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ProviderReply(output=output_type(**reply), model=model, input_tokens=1000, output_tokens=500,
                             cost_usd=0.014)


@pytest.fixture
def run(container: Container) -> Run:
    songs = container.song_repository()
    analysis = songs.analyses.add(SongAnalysis(cache_key="k", audio_sha256="a", lyrics_sha256="l", duration_s=3,
                                               accepted=True, lines=LINES))
    song = songs.songs.add(Song(name="colony", audio_path="a.mp3", lyrics="x", status=SongStatus.READY,
                                analysis_id=analysis.id))
    return container.run_repository().runs.add(Run(song_id=song.id, name="colony", audience=GradeBand.G3_5))


def lesson(container: Container, claude: FakeClaude):
    container.anthropic_service.override(providers.Object(claude))
    return container.lesson_service()


def test_a_plan_is_made_checked_stored_and_costed(container, run):
    plan = lesson(container, FakeClaude(draft())).run(run.id)
    assert plan.video_type == VideoType.CONCEPT and plan.grade_band == GradeBand.G3_5
    assert plan.lines[2].tag == LineTag.UNCLEAR                    # intake's unclear line, set by code
    state = container.run_repository().stage(run.id, StageName.LESSON)
    assert state.status == StageStatus.DONE and state.output["video_type"] == "concept"
    assert container.asset_repository().run_cost(run.id) == pytest.approx(0.014)


def test_a_draft_that_breaks_the_rules_is_re_asked(container, run):
    claude = FakeClaude(draft(tags=("unclear", "number", "illustrate")), draft())
    plan = lesson(container, claude).run(run.id)
    assert claude.calls == 2 and plan.lines[0].tag == LineTag.ILLUSTRATE
    assert container.asset_repository().run_cost(run.id) == pytest.approx(0.028)


def test_the_same_inputs_reuse_the_stored_plan_unless_forced(container, run):
    claude = FakeClaude(draft(), draft(video_type="journey"))
    service = lesson(container, claude)
    service.run(run.id)
    assert service.run(run.id).video_type == VideoType.CONCEPT and claude.calls == 1
    assert service.run(run.id, force=True).video_type == VideoType.JOURNEY and claude.calls == 2


def test_a_requested_type_must_suit_the_band(container, run):
    service = lesson(container, FakeClaude(draft(video_type="journey")))
    assert service.run(run.id, video_type=VideoType.JOURNEY).video_type == VideoType.JOURNEY
    run.audience = GradeBand.K_2
    container.run_repository().runs.update(run)
    with pytest.raises(ValueError, match="not allowed"):
        service.run(run.id, video_type=VideoType.JOURNEY)


def test_a_failure_marks_the_stage_failed(container, run):
    with pytest.raises(Exception, match="refused"):
        lesson(container, FakeClaude(LlmRefusal("no"))).run(run.id)
    state = container.run_repository().stage(run.id, StageName.LESSON)
    assert state.status == StageStatus.FAILED and "refused" in state.error


def test_runs_without_a_band_or_analysis_are_refused(container, run):
    service = lesson(container, FakeClaude())
    run.audience = None
    container.run_repository().runs.update(run)
    with pytest.raises(ValueError, match="grade band"):
        service.run(run.id)
    with pytest.raises(LookupError):
        service.run("missing")


def jamwise_song(container, run, tmp_path):
    """Puts a minimal JamWise export next to the run's song audio."""
    (tmp_path / "jamwise.json").write_text(json.dumps({"result": {"lyrics": "x", "lineTiming": [{}]}}))
    songs = container.song_repository()
    song = songs.songs.get(run.song_id)
    song.audio_path = str(tmp_path / "a.mp3")
    songs.songs.update(song)


def test_lesson_defaults_to_narration_without_jamwise(container, run):
    plan = lesson(container, FakeClaude(draft())).run(run.id)
    assert plan.film_type == FilmType.NARRATION
    assert container.run_repository().stage(run.id, StageName.LESSON).output["film_type"] == "narration"


def test_lesson_defaults_to_music_video_with_jamwise(container, run, tmp_path):
    jamwise_song(container, run, tmp_path)
    claude = FakeClaude(draft())
    assert lesson(container, claude).run(run.id).film_type == FilmType.MUSIC_VIDEO
    assert claude.systems == [JAMWISE_SYSTEM]


def test_lesson_type_narration_on_jamwise_song_uses_narration_prompt(container, run, tmp_path):
    jamwise_song(container, run, tmp_path)
    claude = FakeClaude(draft())
    assert lesson(container, claude).run(run.id, film_type=FilmType.NARRATION).film_type == FilmType.NARRATION
    assert claude.systems == [SYSTEM]


def test_lesson_keeps_type_on_rerun(container, run, tmp_path):
    jamwise_song(container, run, tmp_path)
    claude = FakeClaude(draft(), draft())
    service = lesson(container, claude)
    service.run(run.id, film_type=FilmType.NARRATION)
    assert service.run(run.id, force=True).film_type == FilmType.NARRATION and claude.systems == [SYSTEM, SYSTEM]


def test_old_plan_without_type_is_reused_and_given_one(container, run):
    claude = FakeClaude(draft())
    service = lesson(container, claude)
    service.run(run.id)
    runs = container.run_repository()
    state = runs.stage(run.id, StageName.LESSON)
    state.output = {k: v for k, v in state.output.items() if k != "film_type"}
    runs.stage_states.update(state)
    assert service.run(run.id).film_type == FilmType.NARRATION and claude.calls == 1


def test_realistic_lesson_refuses_2d_run(container, run):
    runs = container.run_repository()
    runs.stage_states.add(StageState(run_id=run.id, stage=StageName.LESSON, status=StageStatus.DONE,
                                     output={"beats": []}))
    claude = FakeClaude(draft())
    with pytest.raises(ValueError, match="is a 2d film"):
        lesson(container, claude).run(run.id)
    assert claude.calls == 0 and runs.stage(run.id, StageName.LESSON).output == {"beats": []}


def test_lesson_refuses_2d_type(container, run):
    with pytest.raises(ValueError, match="run 2d lesson"):
        lesson(container, FakeClaude(draft())).run(run.id, film_type=FilmType.TWO_D)


def test_lesson_refuses_music_video_for_a_song_without_jamwise(container, run):
    claude = FakeClaude(draft())
    with pytest.raises(ValueError, match="music_video needs a JamWise song"):
        lesson(container, claude).run(run.id, film_type=FilmType.MUSIC_VIDEO)
    assert claude.calls == 0
