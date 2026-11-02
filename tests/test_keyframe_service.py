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
from src.schema.storyboard.storyboard_schema import CastKind, CastMember, Scale, Shot, ShotKind, ShotSize
from src.service.keyframe.keyframe_prompt import view_for
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
    assert set(made) == {"s01", "s02", "s03", "s04"} and len(fal.calls) - drawn == 3     # s04 copies s01
    assert made["s04"].id == made["s01"].id
    assets = {a.target_id: a for a in container.asset_repository().assets.list(run_id=run.id, stage=StageName.VISUAL_IDENTITY)}
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
    assert cost == pytest.approx(4 * 0.044 + 4 * 0.001 + 3 * 0.044 + 3 * 0.001)


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
    first, second = sorted(container.asset_repository().assets.list(run_id=run.id, stage=StageName.KEYFRAMES,
                                                                     target_id="s03"), key=lambda a: a.params["try"])
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
    assert made["s01"].review == ReviewState.USED_BEST                                    # a failed item fails the try
    assert "the moment asks for no people beyond the ones listed: two people walk" in made["s01"].review_notes
    assert made["s01"].params["asked_for"][0]["ok"] is False                             # kept on the image
    vision.asked_for = []


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
