import json

import pytest
from dependency_injector import providers

from src.core.container import Container
from src.schema.enums import StageName, StageStatus
from src.schema.lesson.lesson_schema import Beat, GradeBand, LessonPlan
from src.schema.run.run_schema import Run
from src.schema.run.stage_state_schema import StageState
from src.schema.song.song_analysis_schema import LyricLine, SongAnalysis, TimedWord
from src.schema.song.song_schema import Song, SongStatus
from src.service.intake.intake_jamwise import JAMWISE_FILE
from src.service.llm.llm_service import LlmError, ProviderReply
from src.service.storyboard.script_prompt import jamwise_pictures

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
    claude = FakeClaude(drawings(), scripts(), {"gaps": []})
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
    assert "Script:\nb01 bench: show colony center @0; keyword colony @2-4" in claude.prompts[2] and board.gaps == []
    assert container.asset_repository().run_cost(run.id) == pytest.approx(0.15)


def repair(to="colony_huge", **more):
    return {"drawings": [{"id": "colony_huge", "kind": "object", "look": "A huge teal colony", "of": "colony",
                          "motion": "slowly spreads"}],
            "beats": [{"beat": "b02", "background": "bench", "actions": [
                {"verb": "swap", "drawing": "colony", "to": to, "at": 5}]}], **more}


def test_only_the_beats_a_review_names_are_repaired_and_the_script_is_reviewed_again(container, run):
    gap = "b02: the colony is not shown huge; swap colony for a huge colony"
    claude = FakeClaude(drawings(), scripts(), {"gaps": [gap]}, repair(to="nothing"), repair(), {"gaps": []})
    board = storyboard(container, claude).run(run.id)
    assert f"Gaps:\n- {gap}" in claude.prompts[3] and "Script:\nb01 bench: show colony center @0\nb02" in claude.prompts[3]
    assert "b02 swap at 5: to 'nothing' must be a listed" in claude.prompts[4]                # a repair is checked too
    assert "b02 bench: swap colony to colony_huge @5" in claude.prompts[5] and board.gaps == []
    assert [d.id for d in board.drawings] == ["bench", "colony", "dot", "colony_huge"]
    assert [(a.verb, a.to or a.text or a.drawing) for a in board.beats[1].actions] == [("swap", "colony_huge"), ("number", "1858")]
    assert board.beats[0].actions[0].drawing == "colony" and board.beats[2].actions[1].verb == "lipsync"   # the rest is as it was
    assert container.asset_repository().run_cost(run.id) == pytest.approx(0.30)


def test_a_finished_script_is_repaired_again_for_what_a_later_check_finds_in_the_film(container, run):
    claude = FakeClaude(drawings(), scripts(), {"gaps": []}, repair(to="nothing"), repair())
    service = storyboard(container, claude)
    service.run(run.id)
    found = "b02: at 5.4 s the film shows this: the colony is no bigger than before."
    board = service.repair(run.id, [found])
    assert f"Gaps:\n- {found}" in claude.prompts[3] and "b02 bench: swap colony to colony_big @5\n" in claude.prompts[3]
    assert "b02 swap at 5: to 'nothing' must be a listed" in claude.prompts[4]               # checked like any repair
    assert [d.id for d in board.drawings] == ["bench", "colony", "dot", "colony_huge"]
    assert [(a.verb, a.to or a.text or a.drawing) for a in board.beats[1].actions] == [("swap", "colony_huge"), ("number", "1858")]
    assert board.beats[2].actions[1].verb == "lipsync"                                       # code's actions are added afresh
    assert service.run(run.id) == board and claude.replies == []                             # the stage reuses the repaired script
    assert container.asset_repository().run_cost(run.id) == pytest.approx(0.25)
    first = service.run(run.id, force=False).model_copy(update={"gaps": ["kept"]})
    service.restore(run.id, first)
    assert service.run(run.id).gaps == ["kept"]


def test_a_script_gets_two_tries_and_gaps_that_stay_do_not_stop_the_stage(container, run):
    gap, bad = "b02: x", repair(to="nothing")
    claude = FakeClaude(drawings(), scripts(), {"gaps": [gap]}, repair(), {"gaps": [gap, "b03: y"]},     # worse: dropped
                        drawings(), scripts(), {"gaps": [gap]}, repair(), {"gaps": [gap]},              # a tie: the repaired one
                        drawings(), scripts(), {"gaps": [gap]}, bad, bad, bad)                          # no usable repair
    service = storyboard(container, claude)
    board = service.run(run.id)
    assert board.gaps == [gap] and len(claude.prompts) == 5 and "colony_huge" not in [d.id for d in board.drawings]
    assert "colony_huge" in [d.id for d in service.run(run.id, force=True).drawings] and len(claude.prompts) == 10
    again = service.run(run.id, force=True)
    assert again.gaps == [gap] and claude.replies == [] and "colony_huge" not in [d.id for d in again.drawings]
    assert container.run_repository().stage(run.id, StageName.STORYBOARD).status == StageStatus.DONE


def test_the_script_is_given_jamwises_own_pictures(container, run, tmp_path):
    export = {"creativeBrief": {
        "scenes": [{"id": 2, "title": "The Munch", "teaches": "Caterpillars molt to grow", "place": "a giant leaf",
                    "teachingMoment": {"visualDemonstration": "One molt"}}, {"id": 3, "title": "Mood only"}],
        "teachingMoments": [{"sceneId": 2, "visualDemonstration": "Three molts, bigger each time"}],
        "visualWorld": {"chorusVisual": "All four stages", "primarySetting": "A meadow garden"},
        "loopVisual": "The first egg again",
        "numberAnchors": [{"statistic": "4 stages", "visual": "Each lights up"}],
        "sectionPlans": [{"section": "chorus", "keyMoments": ["Identical to opening chorus"]}, {"section": "verse1"}]}}
    text = jamwise_pictures(export)
    for part in ("- Scene 2, The Munch, set in a giant leaf: teaches Caterpillars molt to grow. Shown as: Three molts, "
                 "bigger each time", "- The world: A meadow garden", "- Chorus picture: All four stages", "- 4 stages: Each lights up",
                 "- Section chorus: Identical to opening chorus", "- Ending: The first egg again"):
        assert part in text
    assert "Mood only" not in text and "verse1" not in text and jamwise_pictures({}) is None
    songs = container.song_repository()
    song = songs.songs.get(run.song_id)
    song.audio_path = str(tmp_path / "a.mp3")
    songs.songs.update(song)
    (tmp_path / JAMWISE_FILE).write_text(json.dumps({"result": {"lyrics": "x", "lineTiming": [{}], **export}}))
    claude = FakeClaude(drawings(), scripts(), {"gaps": []})
    storyboard(container, claude).run(run.id)
    assert all("Shown as: Three molts, bigger each time" in p and p.index("Shown as") < p.index("Beats:") for p in claude.prompts)


def test_a_script_that_breaks_the_rules_is_re_asked(container, run):
    claude = FakeClaude(drawings(), scripts(first_show="dish"), scripts(), {"gaps": []})
    board = storyboard(container, claude).run(run.id)
    assert "b01: nothing really moves" in claude.prompts[2] and "b02 swap at 5: colony is not on screen" in claude.prompts[2]
    assert board.beats[0].actions[0].drawing == "colony"
    assert container.asset_repository().run_cost(run.id) == pytest.approx(0.20)


def test_a_script_in_which_one_drawing_hides_another_is_re_asked(container, run):
    stacked = scripts()
    stacked["beats"][2]["actions"][0]["pos"] = "top"            # the singer over the colony at center
    claude = FakeClaude(drawings(), stacked, scripts(), {"gaps": []})
    board = storyboard(container, claude).run(run.id)
    assert "b03: from 8.25 dot covers colony_big; nothing may hide another drawing" in claude.prompts[2]
    assert board.beats[2].actions[0].pos == "left"


def test_later_beats_are_told_what_is_on_screen_and_what_was_scripted(container, run, monkeypatch):
    monkeypatch.setattr("src.service.storyboard.storyboard_service.BEATS_PER_CALL", 2)
    both = scripts()["beats"]
    claude = FakeClaude(drawings(), {"beats": both[:2]}, {"beats": both[2:]}, {"gaps": []})
    storyboard(container, claude).run(run.id)
    assert "Write the scripts for beats b01 to b02." in claude.prompts[1]
    assert "On screen when b03 starts: bench with colony_big at center." in claude.prompts[2]
    assert "b02 bench: swap colony to colony_big @5" in claude.prompts[2]


def test_the_same_inputs_reuse_the_stored_script_unless_forced(container, run):
    claude = FakeClaude(drawings(), scripts(), {"gaps": []}, drawings(spare=False), scripts(), {"gaps": []})
    service = storyboard(container, claude)
    service.run(run.id)
    assert len(service.run(run.id).drawings) == 4 and len(claude.prompts) == 3
    service.run(run.id, force=True)
    assert len(claude.prompts) == 6


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
