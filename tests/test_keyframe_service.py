import pytest
from dependency_injector import providers

from src.core.container import Container
from src.schema.asset.asset_schema import ReviewState
from src.schema.enums import StageName, StageStatus
from src.schema.image.image_check_schema import AskedFor
from src.schema.lesson.lesson_schema import GradeBand
from src.schema.run.run_schema import Run
from src.schema.run.stage_state_schema import StageState
from src.schema.song.song_schema import Song
from src.schema.storyboard.storyboard_schema import CastKind, CastMember, Continuity, Place, Scale, Shot, ShotKind, ShotSize
from src.service.keyframe.keyframe_prompt import keyframe_brief, view_for
from src.service.keyframe.keyframe_service import chains_of
from src.service.storage.storage_service import LocalStorage
from tests.test_sheet_service import BOARD, FakeFal, FakeVision


def shot(id, kind, place, subjects, size, repeat_of=None, lines=(0,)):
    return {"id": id, "slot": id, "start": 0.0, "end": 4.0, "lines": list(lines), "kind": kind, "repeat_of": repeat_of,
            "performer": "host" if kind == "sung" else None, "continuity": "new_scene", "place": place,
            "subjects": subjects, "size": size, "angle": "eye", "position": "left_third",
            "start_state": "The ship sits on the grey sea with its sails full", "action": "The ship sails right",
            "end_state": "The ship is near the centre", "camera": {"move": "static"}, "motion": "low",
            "screen_direction": "none", "transition_in": "cut"}


SHOTS = [shot("s01", "illustrate", "sea", ["ship"], "wide"),
         shot("s02", "sung", "sea", ["host"], "medium_close", lines=(1,)),
         shot("s03", "illustrate", "sea", ["operator", "ship"], "medium", lines=(2,)),
         shot("s04", "illustrate", "sea", ["ship"], "wide", repeat_of="s01", lines=(3,))]


@pytest.fixture
def stage(container: Container, tmp_path):
    fal, vision = FakeFal(), FakeVision()
    container.storage.override(providers.Object(LocalStorage(str(tmp_path))))
    container.fal_image_service.override(providers.Object(fal))
    container.anthropic_service.override(providers.Object(vision))
    container.settings().image_second_opinion_model = None       # on in its own tests below
    sheets = container.sheet_service()
    candidates = sheets.host_candidates(GradeBand.G3_5, count=1)
    sheets.approve_host(GradeBand.G3_5, candidates[0].id)
    return container.keyframe_service(), sheets, fal, vision


def ready_run(container: Container, sheets, operator: str | None = None) -> Run:
    n = len(container.run_repository().runs.list())
    song = container.song_repository().songs.add(Song(name=f"song-{n}", audio_path="a.mp3", lyrics="x"))
    run = container.run_repository().runs.add(Run(song_id=song.id, name="r", audience=GradeBand.G3_5))
    cast = [{**m, "look": operator} if m["id"] == "operator" and operator else m for m in BOARD["cast"]]
    container.run_repository().stage_states.add(StageState(run_id=run.id, stage=StageName.STORYBOARD,
                                                           status=StageStatus.DONE,
                                                           output={**BOARD, "cast": cast, "shots": SHOTS}))
    sheets.run(run.id)
    return run


def test_keyframes_are_drawn_from_the_plate_and_the_right_sheet_views(stage, container):
    service, sheets, fal, vision = stage
    run = ready_run(container, sheets)
    drawn = len(fal.calls)
    made = service.run(run.id)
    assert set(made) == {"s01", "s02", "s03", "s04"} and len(fal.calls) - drawn == 5     # s04 copies s01
    assert made["s04"].id == made["s01"].id
    state = container.run_repository().stage(run.id, StageName.KEYFRAMES)
    assert set(state.output["end_keyframes"]) == {"s01", "s03", "s04"}                  # a sung shot has no ending
    by_id = {a.id: a for a in container.asset_repository().assets.list(run_id=run.id)}
    s01_refs = [by_id[r] for r in made["s01"].params["refs"]]
    assert s01_refs[0].params["view"] == "plate" and s01_refs[0].target_id == "sea"
    assert s01_refs[1].target_id == "ship" and s01_refs[1].params["view"] == "front"
    s03_refs = [by_id[r] for r in made["s03"].params["refs"]]
    assert [r.params["view"] for r in s03_refs] == ["plate", "three_quarter", "front"]        # medium shot: 3/4 view
    kit = {a.params["view"]: a for a in container.asset_repository().assets.list(run_id=None)
           if a.target_id.startswith("host:3-5:explorer:")}
    assert made["s02"].params["refs"] == [s01_refs[0].id, kit["to_camera"].id]         # the sea plate + the outfit face
    assert "The place is image 1" in made["s01"].prompt and "ship is image 2" in made["s01"].prompt.lower()
    rules = next(u for u, labels in vision.seen                 # the judge finds each reference by its label
                 if labels == ["the picture to check", "the place sea", "the cast member ship"])
    assert "labelled 'the place sea'" in rules and "labelled 'the cast member ship'" in rules
    assert "sails full" in made["s01"].prompt and "sails right" not in made["s01"].prompt    # start only, never the action
    assert all(fal.calls[i][1] == "16:9" for i in range(drawn, len(fal.calls)))
    state = container.run_repository().stage(run.id, StageName.KEYFRAMES)
    assert state.status == StageStatus.DONE and state.output["blocked"] == []
    cost = container.asset_repository().run_cost(run.id)
    assert cost == pytest.approx(4 * 0.044 + 4 * 0.001 + 5 * 0.044 + 5 * 0.001)


def test_a_shot_that_fails_every_draw_below_the_floor_fails_the_stage(stage, container):
    service, sheets, fal, vision = stage
    run = ready_run(container, sheets)
    vision.fail_when = ("operator", 2)                           # only s03 has the operator in it; scores 4/10
    made = service.run(run.id)
    assert "s03" not in made and made["s01"].review == ReviewState.PASSED
    tries = container.asset_repository().assets.list(run_id=run.id, stage=StageName.KEYFRAMES, target_id="s03")
    assert sorted(a.params["try"] for a in tries) == [1, 2] and all(a.review == ReviewState.FAILED for a in tries)
    state = container.run_repository().stage(run.id, StageName.KEYFRAMES)
    assert state.status == StageStatus.FAILED and state.output["blocked"] == ["s03"]
    assert "s03: failed every draw; best 4/10" in state.error and "waistcoat" in state.error
    drawn, vision.fail_when = len(fal.calls), None               # a fixed judge passes the kept tries: no redraw
    rechecked = service.recheck(run.id)
    assert len(fal.calls) == drawn and rechecked["s03"].review == ReviewState.PASSED
    assert rechecked["s03"].params["score"] == 9 and rechecked["s03"].review_notes is None
    state = container.run_repository().stage(run.id, StageName.KEYFRAMES)
    assert state.status == StageStatus.DONE and state.error is None and state.output["blocked"] == []
    assert state.output["keyframes"]["s03"] == rechecked["s03"].id


def test_a_shot_that_fails_every_draw_uses_its_best_try_above_the_floor(stage, container):
    service, sheets, fal, vision = stage
    run = ready_run(container, sheets)
    vision.fail_when, vision.fail_score = ("operator", 2), 5     # fails the bar of 6, meets the floor of 5
    made = service.run(run.id)
    assert made["s03"].review == ReviewState.USED_BEST and made["s03"].params["score"] == 5
    assert "waistcoat" in made["s03"].review_notes
    state = container.run_repository().stage(run.id, StageName.KEYFRAMES)
    assert state.status == StageStatus.DONE and state.output["used_best"] == [made["s03"].id]
    assert state.output["blocked"] == [] and state.error is None
    vision.fail_score = 4


def test_a_retry_adds_a_correction_written_from_the_judges_findings(stage, container):
    service, sheets, fal, vision = stage
    run = ready_run(container, sheets)
    vision.fail_when = ("operator", 1)                           # s03 fails once, then passes
    made = service.run(run.id)
    first, second = sorted((a for a in container.asset_repository().assets.list(
        run_id=run.id, stage=StageName.KEYFRAMES, target_id="s03") if a.params["view"] == "keyframe"),
        key=lambda a: a.params["try"])
    assert len(vision.corrections) == 1 and "the waistcoat is blue" in vision.corrections[0]
    assert "sails full" in vision.corrections[0] and first.prompt in vision.corrections[0]   # moment + failed prompt
    assert first.params["correction"] is None and second.params["correction"] == "Show the operator in a red waistcoat."
    assert second.prompt == f"{first.prompt} Show the operator in a red waistcoat." and made["s03"].id == second.id
    drawn = len(fal.calls)
    again = service.run(ready_run(container, sheets).id)        # the corrected retry is found by its seed
    assert len(fal.calls) == drawn and again["s03"].id == second.id


def test_a_thing_in_a_sung_shot_uses_its_front_view():
    sung = Shot.model_construct(kind=ShotKind.SUNG, size=ShotSize.MEDIUM_CLOSE)
    person = CastMember(id="host", kind=CastKind.PERSON, look="a boy", size="a teen", scale=Scale.HUMAN)
    thing = CastMember(id="sun_drawing", kind=CastKind.THING, look="a crayon sun", size="a sheet", scale=Scale.OBJECT)
    assert view_for(person, sung) == "to_camera" and view_for(thing, sung) == "front"


def test_a_seed_is_never_drawn_twice(stage, container):
    service, sheets, fal, vision = stage
    first = ready_run(container, sheets)
    vision.fail_when = ("operator", 1)                           # s03 fails once, passes on its second seed
    service.run(first.id)
    drawn, vision.fail_when = len(fal.calls), None
    made = service.run(ready_run(container, sheets).id)        # same inputs: the passed second try is reused
    assert len(fal.calls) == drawn and made["s03"].review == ReviewState.PASSED and made["s03"].run_id == first.id
    third = ready_run(container, sheets, operator="A Victorian engineer in a tall hat")   # a new look: new seeds
    vision.fail_when, vision.failed_when = ("reference image: operator", 2), 0    # s03 fails both seeds at 4/10
    service.run(third.id)
    drawn, vision.fail_when = len(fal.calls), None
    fourth = ready_run(container, sheets, operator="A Victorian engineer in a tall hat")
    made = service.run(fourth.id)
    state = container.run_repository().stage(fourth.id, StageName.KEYFRAMES)
    assert len(fal.calls) == drawn and "s03" not in made                                 # no new spend, same verdict
    assert state.status == StageStatus.FAILED and "s03: failed every draw" in state.error


def test_keyframes_need_finished_sheets_and_a_shot_without_a_usable_reference_fails(stage, container):
    service, sheets, fal, vision = stage
    n = len(container.run_repository().runs.list())
    song = container.song_repository().songs.add(Song(name=f"song-{n}", audio_path="a.mp3", lyrics="x"))
    run = container.run_repository().runs.add(Run(song_id=song.id, name="r", audience=GradeBand.G3_5))
    container.run_repository().stage_states.add(StageState(run_id=run.id, stage=StageName.STORYBOARD,
                                                           status=StageStatus.DONE, output={**BOARD, "shots": SHOTS}))
    vision.fail_when = ("Victorian", 2)                        # the operator's front view fails both draws
    sheets.run(run.id)
    with pytest.raises(ValueError, match="sheets stage: failed"):   # a failed sheets stage stops the job
        service.run(run.id)
    vision.fail_when = None
    sheets.run(run.id, force=True)
    made = service.run(run.id)
    state = container.run_repository().stage(run.id, StageName.KEYFRAMES)
    assert state.status == StageStatus.DONE and state.output["blocked"] == [] and set(made) == {"s01", "s02", "s03", "s04"}
    sheet_state = container.run_repository().stage(run.id, StageName.VISUAL_IDENTITY)
    plate = container.asset_repository().assets.get(sheet_state.output["sheets"]["sea"]["plate"])
    plate.review = ReviewState.NEEDS_REVIEW
    container.asset_repository().assets.update(plate)
    made = service.run(run.id, force=True)
    state = container.run_repository().stage(run.id, StageName.KEYFRAMES)
    assert state.output["blocked"] == ["s01", "s02", "s03"] and made == {}        # every shot is at sea
    assert state.status == StageStatus.FAILED and "s01: no usable reference" in state.error
    plate.review = ReviewState.APPROVED
    container.asset_repository().assets.update(plate)
    made = service.run(run.id, force=True)
    assert set(made) == {"s01", "s02", "s03", "s04"}


def test_keyframes_need_sheets_first_and_identical_inputs_are_reused(stage, container):
    service, sheets, fal, vision = stage
    n = len(container.run_repository().runs.list())
    song = container.song_repository().songs.add(Song(name=f"song-{n}", audio_path="a.mp3", lyrics="x"))
    bare = container.run_repository().runs.add(Run(song_id=song.id, name="r", audience=GradeBand.G3_5))
    container.run_repository().stage_states.add(StageState(run_id=bare.id, stage=StageName.STORYBOARD,
                                                           status=StageStatus.DONE, output={**BOARD, "shots": SHOTS}))
    with pytest.raises(ValueError, match="no finished reference sheets"):
        service.run(bare.id)
    first = ready_run(container, sheets)
    originals = service.run(first.id)
    drawn, rows = len(fal.calls), len(container.asset_repository().assets.list(run_id=None))
    second = ready_run(container, sheets)                      # same looks: sheets and keyframes both reused
    made = service.run(second.id)
    assert len(fal.calls) == drawn and {s: a.id for s, a in made.items()} == {s: a.id for s, a in originals.items()}
    assert len(container.asset_repository().assets.list(run_id=None)) == rows        # reuse adds no rows
    state = container.run_repository().stage(second.id, StageName.KEYFRAMES)
    assert state.status == StageStatus.DONE and state.output["keyframes"]["s01"] == originals["s01"].id


def test_a_judge_answering_too_many_rules_does_not_fail_the_shot(stage, container):
    service, sheets, fal, vision = stage
    run = ready_run(container, sheets)
    vision.extra_answer = True
    made = service.run(run.id)
    assert all(a.review == ReviewState.PASSED for a in made.values())


def test_the_spend_cap_stops_a_run_before_any_drawing(stage, container):
    service, sheets, fal, vision = stage
    run = ready_run(container, sheets)
    drawn = len(fal.calls)
    container.settings().spend_cap_usd = 0.2                       # the sheets already cost that much
    with pytest.raises(ValueError, match="past the cap"):
        service.run(run.id)
    assert len(fal.calls) == drawn
    container.settings().spend_cap_usd = 25.0


def test_the_stop_loss_pauses_a_run_whose_first_shots_all_fail(stage, container):
    service, sheets, fal, vision = stage
    n = len(container.run_repository().runs.list())
    song = container.song_repository().songs.add(Song(name=f"song-{n}", audio_path="a.mp3", lyrics="x"))
    run = container.run_repository().runs.add(Run(song_id=song.id, name="r", audience=GradeBand.G3_5))
    many = [shot(f"s{i:02d}", "illustrate", "sea", ["ship"], "wide", lines=(i,)) for i in range(1, 21)]
    container.run_repository().stage_states.add(StageState(run_id=run.id, stage=StageName.STORYBOARD,
                                                           status=StageStatus.DONE, output={**BOARD, "shots": many}))
    sheets.run(run.id)
    container.settings().image_workers = 1
    vision.fail_when = ("sails full", 100)                         # every keyframe fails both draws
    made = service.run(run.id)
    state = container.run_repository().stage(run.id, StageName.KEYFRAMES)
    assert state.status == StageStatus.FAILED and len(made) == 0                       # 8 drawn at 4/10, 12 stopped
    assert len(state.output["blocked"]) == 20 and "stopped" in state.output["errors"]["s20"]
    assert "failed every draw" in state.output["errors"]["s01"]
    container.settings().image_workers = 4


def test_the_judge_writes_the_shots_checks_from_its_opening_moment(stage, container):
    service, sheets, fal, vision = stage
    run = ready_run(container, sheets)
    vision.asked_for = [AskedFor(item="no people beyond the ones listed", ok=False, reason="two people walk on the deck")]
    made = service.run(run.id)
    user = next(u for u, labels in vision.seen if labels == ["the picture to check", "the place sea", "the cast member ship"])
    assert "Opening moment:\nThe ship sits on the grey sea" in user and "once each" not in user
    assert made["s01"].review == ReviewState.PASSED and made["s01"].params["try"] == 1  # the score decides
    assert "the moment asks for no people beyond the ones listed: two people walk" in made["s01"].review_notes
    assert made["s01"].params["asked_for"][0]["ok"] is False                             # kept on the image
    vision.asked_for = []


def test_only_the_score_and_the_fatal_rules_fail_a_try(stage, container):
    service, sheets, fal, vision = stage
    vision.fail_rule = "same place as in the image"              # advisory: noted, never a retry
    made = service.run(ready_run(container, sheets).id)
    assert all(a.review == ReviewState.PASSED and a.params["try"] == 1 for a in made.values())
    assert "same place as in the image" in made["s01"].review_notes
    vision.fail_rule = "laid on the picture"                     # fatal: text on the picture fails the try
    made = service.run(ready_run(container, sheets, operator="A sailor in a green scarf").id)   # s03 is new
    tries = [a for a in container.asset_repository().assets.list(run_id=made["s03"].run_id, stage=StageName.KEYFRAMES,
                                                                  target_id="s03") if a.params["view"] == "keyframe"]
    assert len(tries) == 2 and made["s03"].review == ReviewState.USED_BEST                 # both failed; score 9 ≥ floor
    assert any("laid on the picture" in c for c in vision.corrections)                   # the retry was told why
    vision.fail_rule = None


def test_a_judge_reply_that_left_parts_out_is_asked_for_those_only(stage, container):
    service, sheets, fal, vision = stage
    run = ready_run(container, sheets)
    container.settings().image_workers = 1
    vision.drop = 1                                             # the first keyframe check comes back empty
    made = service.run(run.id)
    follow_up = next(u for u, _ in vision.seen if "Your answer left some parts out" in u)
    assert "- the score and its one sentence" in follow_up and "- rule 1. The setting is the same place" in follow_up
    assert all(a.review == ReviewState.PASSED and a.params["try"] == 1 for a in made.values())    # no redraw
    container.settings().image_workers = 4


def test_a_judge_that_never_fills_the_gaps_passes_them_unchecked(stage, container):
    service, sheets, fal, vision = stage
    run = ready_run(container, sheets)
    container.settings().image_workers = 1
    vision.drop = 3                                             # the reply and both follow-ups are empty
    made = service.run(run.id)
    assert all(a.review == ReviewState.PASSED and a.params["try"] == 1 for a in made.values())
    assert sum("Your answer left some parts out" in u for u, _ in vision.seen) == 2       # 3 tries in all
    container.settings().image_workers = 4


def test_shots_in_one_place_are_chained_and_a_still_cut_opens_on_the_previous_ending(stage, container):
    service, sheets, fal, vision = stage
    chained = [shot("s01", "illustrate", "sea", ["ship"], "wide"),
               {**shot("s02", "illustrate", "sea", ["ship"], "wide", lines=(1,)), "continuity": "same_place"},
               {**shot("s03", "illustrate", "sea", ["operator", "ship"], "medium", lines=(2,)), "continuity": "same_place"},
               shot("s04", "sung", "sea", ["host"], "medium_close", lines=(3,))]
    run = ready_run(container, sheets)
    stage_ = container.run_repository().stage(run.id, StageName.STORYBOARD)
    stage_.output = {**stage_.output, "shots": chained}
    container.run_repository().stage_states.update(stage_)
    made = service.run(run.id)
    state = container.run_repository().stage(run.id, StageName.KEYFRAMES)
    ends = {s: container.asset_repository().assets.get(a) for s, a in state.output["end_keyframes"].items()}
    assert made["s02"].id == ends["s01"].id                     # same framing, still camera: the very same picture
    assert made["s03"].params["view"] == "linked_keyframe" and made["s03"].params["refs"][0] == ends["s02"].id
    assert made["s03"].params["labels"][0] == "the previous shot's last frame"
    assert "Image 1 is the last frame of the shot before" in made["s03"].prompt
    assert ends["s03"].params["refs"] == [made["s03"].id] and "Now: The ship is near the centre" in ends["s03"].prompt
    assert made["s04"].params["view"] == "keyframe" and "s04" not in ends                # sung: a new chain, no ending
    assert state.status == StageStatus.DONE and state.output["no_end"] == []


def test_chains_break_at_a_new_scene_a_sung_shot_and_a_repeat():
    def s(id, place="sea", kind=ShotKind.ILLUSTRATE, continuity=Continuity.SAME_PLACE, repeat_of=None):
        return Shot.model_construct(id=id, place=place, kind=kind, continuity=continuity, repeat_of=repeat_of)
    shots = [s("s01", continuity=Continuity.NEW_SCENE), s("s02"), s("s03", kind=ShotKind.SUNG), s("s04"),
             s("s05", repeat_of="s01"), s("s06"), s("s07", place="lab", continuity=Continuity.NEW_SCENE), s("s08", place="lab")]
    assert [[x.id for x in c] for c in chains_of(shots)] == [["s01", "s02"], ["s03"], ["s04"], ["s06"], ["s07", "s08"]]


SECOND = "claude-sonnet-5-5"


def judges(vision, second_score=None):
    """Record which model judged each check; the second judge can be given its own score."""
    models, inspect = [], vision.inspect

    def spy(model, *args):
        models.append(model)
        reply = inspect(model, *args)
        if model == SECOND and second_score is not None:
            reply.output.score = second_score
        return reply
    vision.inspect = spy
    return models


def test_only_a_shot_that_fails_every_draw_gets_the_second_opinion_and_its_pass_stands(stage, container):
    service, sheets, fal, vision = stage
    container.settings().image_second_opinion_model = SECOND
    models = judges(vision)
    run = ready_run(container, sheets)
    vision.fail_when = ("operator", 2)                           # the first judge fails both tries of s03
    made = service.run(run.id)
    assert models.count(SECOND) == 1                             # s03 only; every passing shot stayed with haiku
    assert made["s03"].review == ReviewState.PASSED and made["s03"].params["judged_by"] == SECOND
    assert made["s01"].params["judged_by"] == container.settings().image_check_model
    state = container.run_repository().stage(run.id, StageName.KEYFRAMES)
    assert state.status == StageStatus.DONE and state.output["blocked"] == []


def test_a_second_opinion_that_fails_a_try_decides_the_floor_too(stage, container):
    service, sheets, fal, vision = stage
    container.settings().image_second_opinion_model = SECOND
    judges(vision, second_score=3)
    run = ready_run(container, sheets)
    vision.fail_when, vision.fail_score = ("operator", 2), 5     # haiku: 5, at the floor; sonnet: 3, below it
    made = service.run(run.id)
    assert "s03" not in made
    state = container.run_repository().stage(run.id, StageName.KEYFRAMES)
    assert state.status == StageStatus.FAILED and "s03: failed every draw; best 3/10" in state.error
    vision.fail_score = 4


def test_a_recheck_gives_a_still_failing_shot_the_same_second_opinion(stage, container):
    service, sheets, fal, vision = stage
    run = ready_run(container, sheets)
    vision.fail_when = ("operator", 2)                           # no second opinion yet: s03 fails the stage
    service.run(run.id)
    assert container.run_repository().stage(run.id, StageName.KEYFRAMES).status == StageStatus.FAILED
    container.settings().image_second_opinion_model = SECOND
    models = judges(vision)
    vision.fail_when, vision.failed_when = ("operator", 2), 0    # haiku fails both again on recheck
    rechecked = service.recheck(run.id)
    assert models == [container.settings().image_check_model] * 2 + [SECOND]
    assert rechecked["s03"].review == ReviewState.PASSED and rechecked["s03"].params["judged_by"] == SECOND
    assert container.run_repository().stage(run.id, StageName.KEYFRAMES).status == StageStatus.DONE


def test_a_moment_that_changes_the_place_wins_over_its_usual_look():
    park = Place(id="park", look="a park under a wide open sky", lighting="bright midday sunlight", scale="landscape",
                 walkable=True)
    s = Shot(**shot("s01", "illustrate", "park", [], "wide"))
    brief = keyframe_brief(s, park, [])
    assert "the moment wins" in brief.description and "adds to it or changes in it are fine" in brief.rules[0]
