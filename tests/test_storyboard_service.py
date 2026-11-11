import pytest
from dependency_injector import providers

from src.core.container import Container
from src.schema.enums import StageName, StageStatus
from src.schema.lesson.lesson_schema import Beat, GradeBand, LessonPlan
from src.schema.run.run_schema import Run
from src.schema.run.stage_state_schema import StageState
from src.schema.song.song_analysis_schema import LyricLine, SongAnalysis, TimedWord
from src.schema.song.song_schema import Song, SongStatus
from src.service.llm.llm_service import LlmError, ProviderReply

LINES = [LyricLine(index=i, text=t, start=i * 4, end=i * 4 + 3.8, confidence=0.9)
         for i, t in enumerate(["This is a colony.", "In 1858 it grew.", "Sing it with me."])]
WORDS = [TimedWord(index=0, line_index=0, text="colony.", start=2.0, end=2.5),
         TimedWord(index=1, line_index=1, text="1858", start=5.0, end=5.5)]
BEATS = [Beat(id="b01", start=0, end=4, type="explain", lines=[0], concept="A colony is a group of bacteria.",
              keyword="colony", keyword_at=2.0, hint="A dish with a colony"),
         Beat(id="b02", start=4, end=8, type="example", lines=[1], concept="A colony grows.", number_text="1858",
              hint="The colony spreads", avoid="Do not show one giant germ"),
         Beat(id="b03", start=8, end=12, type="hero", lines=[2], concept="Colonies grow.", hint="The singer sings")]


def drawings(spare=True):
    return {"performer": "Dot, a round teal bacterium", "palette": ["#12A594", "#F2F2F2", "#FFC857", "#222222"],
            "drawings": [{"id": "bench", "kind": "background", "look": "A white lab bench"},
                         {"id": "dish", "kind": "object", "look": "A clear glass dish"},
                         {"id": "colony", "kind": "object", "look": "A small teal colony", "motion": "slowly spreads"},
                         {"id": "colony_big", "kind": "object", "look": "A large teal colony", "of": "colony",
                          "motion": "slowly spreads"},
                         {"id": "dot", "kind": "performer", "look": "Arms wide", "motion": "sways", "sings": True}]
            + ([{"id": "arrow", "kind": "icon", "look": "A curved arrow"}] if spare else [])}


def scripts(first_show="colony"):
    return {"beats": [
        {"beat": "b01", "background": "bench", "actions": [{"verb": "show", "drawing": first_show, "pos": "center", "at": 0}]},
        {"beat": "b02", "background": "bench", "actions": [{"verb": "swap", "drawing": "colony", "to": "colony_big", "at": 5}]},
        {"beat": "b03", "background": "bench", "actions": [{"verb": "show", "drawing": "dot", "pos": "left", "at": 8}]}]}


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
                                               accepted=True, lines=LINES, words=WORDS))
    song = songs.songs.add(Song(name="colony", audio_path="a.mp3", lyrics="x", status=SongStatus.READY,
                                analysis_id=analysis.id))
    run = runs.runs.add(Run(song_id=song.id, name="colony", audience=GradeBand.G3_5))
    plan = LessonPlan(topic="Bacteria", subject="Science", summary="S.", key_terms=[], must_not_teach=[], beats=BEATS,
                      grade_band=GradeBand.G3_5, model="m", prompt_version="v")
    runs.stage_states.add(StageState(run_id=run.id, stage=StageName.LESSON, status=StageStatus.DONE,
                                     output=plan.model_dump(mode="json")))
    return run


def storyboard(container: Container, claude: FakeClaude):
    container.anthropic_service.override(providers.Object(claude))
    return container.storyboard_service()


def test_a_script_is_written_checked_finished_stored_and_costed(container, run):
    claude = FakeClaude(drawings(), scripts())
    board = storyboard(container, claude).run(run.id)
    assert board.style.startswith("A bold 2D cartoon") and board.performer == "Dot, a round teal bacterium"
    assert [d.id for d in board.drawings] == ["bench", "colony", "colony_big", "dot"]       # unused ones are dropped
    assert [(b.beat, b.start, b.end, b.type) for b in board.beats] == [
        ("b01", 0, 4, "explain"), ("b02", 4, 8, "example"), ("b03", 8, 12, "hero")]
    assert [(a.verb, a.text or a.drawing, a.at) for b in board.beats for a in b.actions] == [
        ("show", "colony", 0), ("keyword", "colony", 2.0), ("swap", "colony", 5), ("number", "1858", 5.0),
        ("show", "dot", 8), ("lipsync", "dot", 8)]
    assert 'b01 0-4s explain: A colony is a group of bacteria.\n    Key word "colony" sung at 2s' in claude.prompts[0]
    assert 'Figure our code draws: "1858"' in claude.prompts[0] and "Sung from 4s: In 1858 it grew." in claude.prompts[0]
    assert "Avoid: Do not show one giant germ" in claude.prompts[0]
    assert "- colony (object, motion: slowly spreads): A small teal colony" in claude.prompts[1]
    assert "On screen when b01 starts: nothing." in claude.prompts[1]
    state = container.run_repository().stage(run.id, StageName.STORYBOARD)
    assert state.status == StageStatus.DONE and state.output["beats"][2]["actions"][1]["verb"] == "lipsync"
    assert container.asset_repository().run_cost(run.id) == pytest.approx(0.10)


def test_a_script_that_breaks_the_rules_is_re_asked(container, run):
    claude = FakeClaude(drawings(), scripts(first_show="dish"), scripts())
    board = storyboard(container, claude).run(run.id)
    assert "b01: nothing really moves" in claude.prompts[2] and "b02 swap at 5: colony is not on screen" in claude.prompts[2]
    assert board.beats[0].actions[0].drawing == "colony"
    assert container.asset_repository().run_cost(run.id) == pytest.approx(0.15)


def test_later_beats_are_told_what_is_on_screen_and_what_was_scripted(container, run, monkeypatch):
    monkeypatch.setattr("src.service.storyboard.storyboard_service.BEATS_PER_CALL", 2)
    both = scripts()["beats"]
    claude = FakeClaude(drawings(), {"beats": both[:2]}, {"beats": both[2:]})
    storyboard(container, claude).run(run.id)
    assert "Write the scripts for beats b01 to b02." in claude.prompts[1]
    assert "On screen when b03 starts: bench with colony_big at center." in claude.prompts[2]
    assert "b02 bench: swap colony to colony_big @5" in claude.prompts[2]


def test_the_same_inputs_reuse_the_stored_script_unless_forced(container, run):
    claude = FakeClaude(drawings(), scripts(), drawings(spare=False), scripts())
    service = storyboard(container, claude)
    service.run(run.id)
    assert len(service.run(run.id).drawings) == 4 and len(claude.prompts) == 2
    service.run(run.id, force=True)
    assert len(claude.prompts) == 4


def test_a_script_that_never_passes_fails_the_stage_and_keeps_what_was_written(container, run):
    bad = scripts(first_show="dish")
    with pytest.raises(LlmError, match="nothing really moves"):
        storyboard(container, FakeClaude(drawings(), bad, bad, bad)).run(run.id)
    state = container.run_repository().stage(run.id, StageName.STORYBOARD)
    assert state.status == StageStatus.FAILED and len(state.output["drawings"]["drawings"]) == 6
    assert container.asset_repository().run_cost(run.id) == pytest.approx(0.20)


def test_a_run_needs_a_lesson_plan_that_matches_its_song(container, run):
    runs, songs = container.run_repository(), container.song_repository()
    analysis = songs.analyses.get(songs.songs.get(run.song_id).analysis_id)
    analysis.lines = LINES[:2]
    songs.analyses.update(analysis)
    with pytest.raises(ValueError, match="older than its song analysis"):
        storyboard(container, FakeClaude()).run(run.id)
    other = runs.runs.add(Run(song_id=run.song_id, name="other", audience=GradeBand.G3_5))
    with pytest.raises(ValueError, match="no lesson plan"):
        container.storyboard_service().run(other.id)
