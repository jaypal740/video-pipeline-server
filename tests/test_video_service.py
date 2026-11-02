from pathlib import Path

import pytest
from dependency_injector import providers

from src.core.container import Container
from src.schema.asset.asset_schema import ReviewState
from src.schema.enums import StageName, StageStatus
from src.schema.storyboard.storyboard_schema import Storyboard
from src.service.storage.storage_service import LocalStorage
from src.service.video.clip_checks import ClipChecks
from src.service.video.fal_video_service import VideoReply
from tests.test_keyframe_service import ready_run
from tests.test_sheet_service import FakeFal, FakeVision, tiny_png


class FakeFalVideo:
    """Returns a different fake clip for every (prompt, seed); records what it drew."""

    name = "fal"

    def __init__(self):
        self.calls = []

    def generate(self, model, prompt, image_url, seconds, seed=None, negative=None, end_image_url=None):
        self.calls.append((model, image_url, seconds, seed, prompt, end_image_url))
        return VideoReply(data=f"mp4|{prompt}|{seed}|audio".encode(), content_type="video/mp4", model=model, seed=seed,
                          seconds_requested=seconds, cost_usd=round(0.084 * seconds, 3), seconds=90.0,
                          request_id=f"req-{len(self.calls)}")

    def upload(self, data, content_type):
        return f"https://fal.fake/{len(data)}"


class FakeClipChecks:
    """Code checks that pass unless `fail_when` names a shot: then the first N clips of it are stills."""

    def __init__(self):
        self.fail_when, self.failed = None, 0

    def check(self, data, wanted_seconds):
        still = self.fail_when is not None and self.fail_when[0] in data.decode() and self.failed < self.fail_when[1]
        if still:
            self.failed += 1
        return ClipChecks(seconds=float(wanted_seconds), fps=24.0, width=1280, height=720, motion=0.1 if still else 5.0,
                          tail_motion=2.0, black_frames=0,
                          problems=["almost nothing moves (motion 0.10); the picture reads as a still"] if still else [])

    def frames(self, data, seconds):
        return [tiny_png(1, "16:9"), tiny_png(2, "16:9"), tiny_png(3, "16:9")]


class FakeFfmpeg:
    """Writes the output file of the two calls the video stage makes: dropping audio, and a camera move."""

    def __init__(self):
        self.calls = []

    def run(self, args):
        self.calls.append(args)
        out = Path(args[-1])
        if "-an" in args:
            out.write_bytes(Path(args[args.index("-i") + 1]).read_bytes().replace(b"|audio", b""))
        else:
            out.write_bytes(f"mp4|camera|{args[args.index('-vf') + 1]}".encode())
        return b""


@pytest.fixture
def stage(container: Container, tmp_path):
    fal, vision, video, checks, ffmpeg = FakeFal(), FakeVision(), FakeFalVideo(), FakeClipChecks(), FakeFfmpeg()
    container.storage.override(providers.Object(LocalStorage(str(tmp_path))))
    container.fal_image_service.override(providers.Object(fal))
    container.anthropic_service.override(providers.Object(vision))
    container.fal_video_service.override(providers.Object(video))
    container.clip_check_service.override(providers.Object(checks))
    container.ffmpeg_service.override(providers.Object(ffmpeg))
    sheets = container.sheet_service()
    sheets.approve_host("3-5", sheets.host_candidates("3-5", count=1)[0].id)
    return container.video_service(), sheets, video, checks, vision


def drawn_run(container, sheets):
    run = ready_run(container, sheets)
    container.keyframe_service().run(run.id)
    return run


def test_clips_are_drawn_between_the_start_and_end_keyframes_and_repeats_reuse_them(stage, container):
    service, sheets, video, checks, vision = stage
    run = drawn_run(container, sheets)
    made = service.run(run.id)
    assert set(made) == {"s01", "s03", "s04"} and made["s04"].id == made["s01"].id      # s02 is sung; s04 repeats s01
    assert len(video.calls) == 2 and all(a.review == ReviewState.PASSED for a in made.values())
    model, url, seconds, seed, prompt, end_url = video.calls[0]
    assert model == "minimax-h3-max-turbo" and url.startswith("https://fal.fake/") and seconds == 4.0   # the slot itself
    output = container.run_repository().stage(run.id, StageName.KEYFRAMES).output
    assert all(c[5] is not None for c in video.calls)            # every clip is sent its end frame
    assert made["s01"].params["keyframe"] == output["keyframes"]["s01"] and made["s01"].params["route"] == "model"
    assert made["s01"].params["end_keyframe"] == output["end_keyframes"]["s01"]
    assert made["s01"].params["request_id"].startswith("req-")
    assert made["s01"].params["score"] == 9 and made["s01"].params["motion"] == 5.0
    stored = container.storage().get(made["s01"].path)
    assert b"audio" not in stored and stored.startswith(b"mp4|")  # the made-up audio is dropped before storing
    state = container.run_repository().stage(run.id, StageName.VIDEO)
    assert state.status == StageStatus.DONE and state.output["blocked"] == [] and state.output["used_best"] == []
    judged = [u for u, labels in vision.seen if "frame 1 of 3, the first" in labels]
    assert judged and "first, middle and last frames of one short clip" in judged[0]
    drawn = len(video.calls)
    again = service.run(run.id)                                 # inputs unchanged: nothing drawn
    assert len(video.calls) == drawn and again["s01"].id == made["s01"].id


def test_a_still_clip_fails_the_code_checks_and_the_retry_carries_the_finding(stage, container):
    service, sheets, video, checks, vision = stage
    run = drawn_run(container, sheets)
    checks.fail_when = ("sails right", 1)                        # s01 and s03 share the action; the first is a still
    made = service.run(run.id)
    tries = sorted(container.asset_repository().assets.list(run_id=run.id, stage=StageName.VIDEO, target_id="s01")
                   + container.asset_repository().assets.list(run_id=run.id, stage=StageName.VIDEO, target_id="s03"),
                   key=lambda a: (a.target_id, a.params["try"]))
    failed = [a for a in tries if a.review == ReviewState.FAILED]
    assert len(failed) == 1 and failed[0].params["score"] is None and "nothing moves" in failed[0].review_notes
    retry = next(a for a in tries if a.target_id == failed[0].target_id and a.params["try"] == 2)
    assert retry.review == ReviewState.PASSED and retry.params["correction"] == "Show the operator in a red waistcoat."
    assert any("nothing moves" in c for c in vision.corrections)
    assert all(a.review == ReviewState.PASSED for a in made.values())


def test_a_clip_that_fails_the_judge_uses_its_best_try_or_fails_the_stage(stage, container):
    service, sheets, video, checks, vision = stage
    run = drawn_run(container, sheets)
    vision.fail_when, vision.fail_score = ("frames of one short clip", 100), 5     # every clip judged 5/10
    made = service.run(run.id)
    assert all(a.review == ReviewState.USED_BEST and a.params["try"] == 1 for a in made.values())
    state = container.run_repository().stage(run.id, StageName.VIDEO)
    assert state.status == StageStatus.DONE and len(state.output["used_best"]) == 2
    vision.fail_score, vision.failed_when = 4, 0
    container.settings().video_length_factor = 1.2              # new lengths: new seeds, nothing cached
    run = drawn_run(container, sheets)
    made = service.run(run.id)
    state = container.run_repository().stage(run.id, StageName.VIDEO)
    assert state.status == StageStatus.FAILED and "s01: failed every draw; best 4/10" in state.error
    assert "s01" not in made
    vision.fail_when, container.settings().video_length_factor = None, 1.0
    drawn = len(video.calls)
    rechecked = service.recheck(run.id)                          # a fixed judge passes the kept tries: no redraw
    assert len(video.calls) == drawn and rechecked["s01"].review == ReviewState.PASSED
    state = container.run_repository().stage(run.id, StageName.VIDEO)
    assert state.status == StageStatus.DONE and state.error is None and state.output["clips"]["s01"] == rechecked["s01"].id


def test_sung_shots_are_left_to_lip_sync_and_the_cap_stops_a_run_before_drawing(stage, container):
    service, sheets, video, checks, vision = stage
    run = drawn_run(container, sheets)
    made = service.run(run.id)
    assert "s02" not in made and len(video.calls) == 2           # the default: lip-sync makes the sung shot
    container.settings().video_sung_mode = "calm"
    container.settings().spend_cap_usd = 0.3
    with pytest.raises(ValueError, match="past the cap"):         # only the calm sung clip is new to draw
        service.run(drawn_run(container, sheets).id)
    container.settings().spend_cap_usd, container.settings().video_sung_mode = 25.0, "skip"
    with pytest.raises(ValueError, match="no finished keyframes"):
        service.run(ready_run(container, sheets).id)


def test_a_shot_where_only_the_camera_moves_is_an_ffmpeg_move_over_its_keyframe(stage, container):
    service, sheets, video, checks, vision = stage
    run = drawn_run(container, sheets)
    runs = container.run_repository()
    state = runs.stage(run.id, StageName.STORYBOARD)
    board = Storyboard(**state.output)
    s03 = next(s for s in board.shots if s.id == "s03")
    s03.moves, s03.camera.move, s03.camera.direction = "camera", "pan", "right"
    state.output = board.model_dump(mode="json")
    runs.stage_states.update(state)
    made = service.run(run.id)
    assert made["s03"].params["route"] == "camera" and made["s03"].provider == "ffmpeg" and made["s03"].cost_usd == 0
    assert made["s03"].params["frames"] == 96 and len(video.calls) == 1            # 4 s at 24 fps; only s01 drawn
    assert b"zoompan" in container.storage().get(made["s03"].path)
    again = service.run(run.id, force=True)                     # the same move on the same keyframe is made once
    assert again["s03"].id == made["s03"].id
