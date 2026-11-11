import pytest
from dependency_injector import providers

from src.core.container import Container
from src.schema.enums import StageName, StageStatus
from src.schema.lesson.lesson_schema import GradeBand
from src.schema.run.run_schema import Run
from src.schema.song.song_analysis_schema import LyricLine, SongAnalysis, TimedWord
from src.schema.song.song_schema import Song, SongStatus
from src.service.llm.llm_service import LlmRefusal, ProviderReply

LINES = [LyricLine(index=i, text=t, start=i, end=i + 1, confidence=0.9, unclear=(i == 2))
         for i, t in enumerate(["This is a colony.", "In 1858 it grew.", "Mumble words."])]


WORDS = [TimedWord(index=i, line_index=0, text=t, start=i * 0.25, end=i * 0.25 + 0.2)
         for i, t in enumerate(["This", "is", "a", "colony."])]


def draft(keyword="colony", topic="Bacteria"):
    return {"topic": topic, "subject": "Science", "summary": "S.",
            "key_terms": [{"term": "colony", "definition": "a group"}], "must_not_teach": [],
            "beats": [{"first_line": 0, "last_line": 1, "type": "explain", "concept": "A colony is a group of bacteria.",
                       "keyword": keyword, "number_text": "1858", "hint": "A dish with a colony spreading"},
                      {"first_line": 2, "last_line": 2, "type": "example", "concept": "Colonies keep growing.",
                       "hint": "The colony fills the dish"}]}


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
                                               accepted=True, lines=LINES, words=WORDS))
    song = songs.songs.add(Song(name="colony", audio_path="a.mp3", lyrics="x", status=SongStatus.READY,
                                analysis_id=analysis.id))
    return container.run_repository().runs.add(Run(song_id=song.id, name="colony", audience=GradeBand.G3_5))


def lesson(container: Container, claude: FakeClaude):
    container.anthropic_service.override(providers.Object(claude))
    return container.lesson_service()


def test_a_plan_is_made_checked_set_in_time_stored_and_costed(container, run):
    plan = lesson(container, FakeClaude(draft())).run(run.id)
    assert plan.grade_band == GradeBand.G3_5
    assert [(b.id, b.start, b.end, b.lines) for b in plan.beats] == [("b01", 0.0, 2.0, [0, 1]), ("b02", 2.0, 3.0, [2])]
    assert (plan.beats[0].keyword, plan.beats[0].keyword_at) == ("colony", 0.75)     # when the word is sung
    state = container.run_repository().stage(run.id, StageName.LESSON)
    assert state.status == StageStatus.DONE and state.output["beats"][0]["number_text"] == "1858"
    assert container.asset_repository().run_cost(run.id) == pytest.approx(0.014)


def test_a_draft_that_breaks_the_rules_is_re_asked(container, run):
    claude = FakeClaude(draft(keyword="bacteria"), draft())
    plan = lesson(container, claude).run(run.id)
    assert claude.calls == 2 and plan.beats[0].keyword == "colony"
    assert container.asset_repository().run_cost(run.id) == pytest.approx(0.028)


def test_the_same_inputs_reuse_the_stored_plan_unless_forced(container, run):
    claude = FakeClaude(draft(), draft(topic="Germs"))
    service = lesson(container, claude)
    service.run(run.id)
    assert service.run(run.id).topic == "Bacteria" and claude.calls == 1
    assert service.run(run.id, force=True).topic == "Germs" and claude.calls == 2


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
