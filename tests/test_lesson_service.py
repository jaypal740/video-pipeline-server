import pytest
from dependency_injector import providers

from src.core.container import Container
from src.schema.enums import StageName, StageStatus
from src.schema.lesson.lesson_schema import GradeBand, LineTag, VideoType
from src.schema.run.run_schema import Run
from src.schema.song.song_analysis_schema import LyricLine, SongAnalysis
from src.schema.song.song_schema import Song, SongStatus
from src.service.llm.llm_service import LlmRefusal, ProviderReply

LINES = [LyricLine(index=i, text=t, start=i, end=i + 1, confidence=0.9, unclear=(i == 2))
         for i, t in enumerate(["This is a colony.", "In 1858 it grew.", "Mumble words."])]


def draft(video_type="concept", tags=("illustrate", "number", "illustrate")):
    return {"topic": "Bacteria", "subject": "Science", "summary": "S.", "video_type": video_type,
            "type_reason": "r", "key_terms": [{"term": "colony", "definition": "a group"}], "must_not_teach": [],
            "lines": [{"index": i, "teaches": "t", "show": "s", "tag": tag,
                       "number_text": "1858" if tag == "number" else None} for i, tag in enumerate(tags)]}


class FakeClaude:
    name = "anthropic"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    def parse(self, model, system, user, output_type, effort):
        self.calls += 1
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
