import pytest
from dependency_injector import providers

from src.core.container import Container
from src.schema.asset.asset_schema import ReviewState
from src.schema.enums import StageName, StageStatus
from src.schema.lesson.lesson_schema import GradeBand
from src.schema.run.run_schema import Run
from src.schema.run.stage_state_schema import StageState
from src.schema.song.song_schema import Song
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


def ready_run(container: Container, sheets) -> Run:
    n = len(container.run_repository().runs.list())
    song = container.song_repository().songs.add(Song(name=f"song-{n}", audio_path="a.mp3", lyrics="x"))
    run = container.run_repository().runs.add(Run(song_id=song.id, name="r", audience=GradeBand.G3_5))
    container.run_repository().stage_states.add(StageState(run_id=run.id, stage=StageName.STORYBOARD,
                                                           status=StageStatus.DONE, output={**BOARD, "shots": SHOTS}))
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
    assert made["s02"].params["refs"] == [s01_refs[0].id, kit["to_camera"].id]         # the sea plate + the kit face
    assert "The place is image 1" in made["s01"].prompt and "ship is image 2" in made["s01"].prompt.lower()
    assert "sails full" in made["s01"].prompt and "sails right" not in made["s01"].prompt    # start only, never the action
    assert all(fal.calls[i][1] == "16:9" for i in range(drawn, len(fal.calls)))
    state = container.run_repository().stage(run.id, StageName.KEYFRAMES)
    assert state.status == StageStatus.DONE and state.output["blocked"] == []
    cost = container.asset_repository().run_cost(run.id)
    assert cost == pytest.approx(4 * 0.044 + 4 * 0.001 + 3 * 0.044 + 3 * 0.001)


def test_a_failing_keyframe_is_redrawn_then_left_for_review(stage, container):
    service, sheets, fal, vision = stage
    run = ready_run(container, sheets)
    vision.fail_when = ("operator", 3)                           # only s03 has the operator in it
    made = service.run(run.id)
    waiting = [a for a in made.values() if a.review == ReviewState.NEEDS_REVIEW]
    assert len(waiting) == 1 and waiting[0].target_id == "s03" and waiting[0].params["try"] == 3
    assert "/3-" in waiting[0].path and made["s01"].review == ReviewState.PASSED
    tries = container.asset_repository().assets.list(run_id=run.id, stage=StageName.KEYFRAMES,
                                                     target_id=waiting[0].target_id)
    assert sorted(a.params["try"] for a in tries) == [1, 2, 3]
    state = container.run_repository().stage(run.id, StageName.KEYFRAMES)
    assert state.status == StageStatus.WAITING_GATE and state.output["needs_review"] == [waiting[0].id]


def test_a_shot_whose_sheet_needs_review_is_blocked_until_a_person_approves_it(stage, container):
    service, sheets, fal, vision = stage
    n = len(container.run_repository().runs.list())
    song = container.song_repository().songs.add(Song(name=f"song-{n}", audio_path="a.mp3", lyrics="x"))
    run = container.run_repository().runs.add(Run(song_id=song.id, name="r", audience=GradeBand.G3_5))
    container.run_repository().stage_states.add(StageState(run_id=run.id, stage=StageName.STORYBOARD,
                                                           status=StageStatus.DONE, output={**BOARD, "shots": SHOTS}))
    vision.fail_when = ("Victorian", 3)                        # the operator's front view fails three times
    sheets.run(run.id)
    made = service.run(run.id)
    state = container.run_repository().stage(run.id, StageName.KEYFRAMES)
    assert state.status == StageStatus.DONE and state.output["blocked"] == [] and set(made) == {"s01", "s02", "s03", "s04"}
    # the operator's three-quarter view passed, so s03 could be drawn; make the sea plate unusable and its shots block
    plate = [a for a in container.asset_repository().assets.list(run_id=run.id, target_id="sea")][0]
    plate.review = ReviewState.NEEDS_REVIEW
    container.asset_repository().assets.update(plate)
    made = service.run(run.id, force=True)
    state = container.run_repository().stage(run.id, StageName.KEYFRAMES)
    assert state.output["blocked"] == ["s01", "s02", "s03"] and made == {}        # every shot is at sea
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
    with pytest.raises(ValueError, match="no reference sheets"):
        service.run(bare.id)
    first = ready_run(container, sheets)
    service.run(first.id)
    drawn = len(fal.calls)
    second = ready_run(container, sheets)                      # same looks: sheets and keyframes both reused
    made = service.run(second.id)
    assert len(fal.calls) == drawn and all(a.cost_usd == 0.0 for a in made.values())
