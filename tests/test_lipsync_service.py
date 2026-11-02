from pathlib import Path

import pytest
from dependency_injector import providers

from src.core.container import Container
from src.schema.asset.asset_schema import ReviewState
from src.schema.enums import StageName, StageStatus
from src.service.lipsync.fal_lipsync_service import LipsyncReply
from src.service.lipsync.lipsync_service import KEYFRAME_LABEL, LipsyncService
from src.service.storage.storage_service import LocalStorage
from src.service.video.clip_checks import ClipChecks
from tests.test_keyframe_service import ready_run
from tests.test_sheet_service import FakeFal, FakeVision, tiny_png


class FakeFalLipsync:
    """Returns a different fake clip, with audio, for every (seed, audio); records what it drew."""

    name = "fal"

    def __init__(self):
        self.calls, self.uploads = [], []

    def generate(self, model, image_url, audio_url, audio_seconds, seed=None):
        self.calls.append((model, image_url, audio_url, audio_seconds, seed))
        return LipsyncReply(data=f"mp4+audio|{seed}|{audio_url}".encode(), content_type="video/mp4", model=model,
                            seed=seed, audio_seconds=audio_seconds, cost_usd=round(0.08 * audio_seconds, 4),
                            seconds=20.0, request_id=f"req-{len(self.calls)}")

    def upload(self, data, content_type):
        self.uploads.append((data, content_type))
        return f"https://fal.fake/{content_type}/{len(self.uploads)}"


class FakeFfmpeg:
    """Writes the output file a call names: an audio cut records where it starts and how long it is; stripping
    the audio marks the clip silent."""

    def __init__(self):
        self.calls = []

    def run(self, args):
        self.calls.append(args)
        out = Path(args[-1])
        if "-ss" in args:
            out.write_bytes(f"aac|{args[args.index('-ss') + 1]}|{args[args.index('-t') + 1]}".encode())
        else:
            out.write_bytes(b"silent|" + Path(args[args.index("-i") + 1]).read_bytes())
        return b""


class FakeClipChecks:
    """Every clip barely moves (a singing face), which must not fail it; `black` gives the next N clips
    black frames."""

    def __init__(self):
        self.black, self.seen = 0, []

    def check(self, data, wanted_seconds):
        self.seen.append(data)
        black = 3 if self.black > 0 else 0
        self.black -= 1
        return ClipChecks(seconds=5.0, fps=25.0, width=1366, height=768, motion=0.1, tail_motion=0.1,
                          black_frames=black, problems=["almost nothing moves (motion 0.10); the picture reads as a still"])

    def frames(self, data, seconds):
        return [tiny_png(1, "16:9"), tiny_png(2, "16:9"), tiny_png(3, "16:9")]


@pytest.fixture
def stage(container: Container, tmp_path):
    fal, vision, lipsync, ffmpeg, checks = FakeFal(), FakeVision(), FakeFalLipsync(), FakeFfmpeg(), FakeClipChecks()
    container.storage.override(providers.Object(LocalStorage(str(tmp_path))))
    container.fal_image_service.override(providers.Object(fal))
    container.anthropic_service.override(providers.Object(vision))
    sheets = container.sheet_service()
    sheets.approve_host("3-5", sheets.host_candidates("3-5", count=1)[0].id)
    service = LipsyncService(run_repository=container.run_repository(), song_repository=container.song_repository(),
                             asset_repository=container.asset_repository(), storage=container.storage(),
                             lipsync=lipsync, ffmpeg=ffmpeg, checks=checks, check=container.image_check_service(),
                             settings=container.settings())
    return service, sheets, lipsync, ffmpeg, checks, vision


def drawn_run(container, sheets, sung_repeat=False):
    run = ready_run(container, sheets)
    container.keyframe_service().run(run.id)
    if sung_repeat:                                             # s02 sings 4-8 s; s04 repeats it as a chorus
        state = container.run_repository().stage(run.id, StageName.STORYBOARD)
        moved = {"s02": {"start": 4.0, "end": 8.0}, "s04": {"kind": "sung", "repeat_of": "s02"}}
        shots = [{**s, **moved.get(s["id"], {})} for s in state.output["shots"]]
        state.output = {**state.output, "shots": shots}
        container.run_repository().stage_states.update(state)
    return run


def test_a_sung_shot_is_lip_synced_from_its_keyframe_and_the_song_from_its_start(stage, container):
    service, sheets, lipsync, ffmpeg, checks, vision = stage
    run = drawn_run(container, sheets, sung_repeat=True)
    made = service.run(run.id)
    assert set(made) == {"s02", "s04"} and made["s04"].id == made["s02"].id and len(lipsync.calls) == 1
    model, image_url, audio_url, audio_seconds, seed = lipsync.calls[0]
    assert model == "minimax-h3-max-lipsync" and audio_seconds == 5.0                  # a 4 s slot sends 5 s
    assert image_url.startswith("https://fal.fake/image/png") and audio_url.startswith("https://fal.fake/audio/mp4")
    cut = ffmpeg.calls[0]
    assert cut[cut.index("-ss") + 1] == "4.000" and cut[cut.index("-t") + 1] == "5.000" and "a.mp3" in cut
    clip = made["s02"]
    assert clip.stage == StageName.LIPSYNC and clip.review == ReviewState.PASSED
    assert clip.path.startswith(f"runs/{run.id}/lipsync/s02/1-")
    assert container.storage().get(clip.path).startswith(b"silent|") and checks.seen[0].startswith(b"silent|")
    keyframes = container.run_repository().stage(run.id, StageName.KEYFRAMES).output["keyframes"]
    assert clip.params["keyframe"] == keyframes["s02"] and clip.params["audio_start"] == 4.0
    assert clip.params["audio_seconds"] == 5.0 and clip.params["seconds"] == 5.0 and clip.params["score"] == 9
    assert clip.params["checks"] == [] and clip.params["request_id"] == "req-1"       # the motion rule is not used
    judged = [(u, labels) for u, labels in vision.seen if "singing his line" in u]
    assert judged and judged[0][1] == ["the picture to check", KEYFRAME_LABEL]
    costs = [c for c in container.asset_repository().cost_entries.list(run_id=run.id) if c.stage == StageName.LIPSYNC]
    assert {(c.unit, c.units) for c in costs if c.unit == "second"} == {("second", 5.0)}
    assert sum(c.cost_usd for c in costs) == pytest.approx(0.4 + 0.001)
    state = container.run_repository().stage(run.id, StageName.LIPSYNC)
    assert state.status == StageStatus.DONE and state.output == {
        "clips": {"s02": clip.id, "s04": clip.id}, "used_best": [], "blocked": [], "errors": {}}
    again = service.run(run.id)                                 # inputs unchanged: nothing drawn
    assert len(lipsync.calls) == 1 and again["s02"].id == clip.id


def test_black_frames_fail_a_try_and_the_retry_uses_a_new_seed(stage, container):
    service, sheets, lipsync, ffmpeg, checks, vision = stage
    run = drawn_run(container, sheets)
    checks.black = 1
    made = service.run(run.id)
    tries = sorted(container.asset_repository().assets.list(run_id=run.id, stage=StageName.LIPSYNC, target_id="s02"),
                   key=lambda a: a.params["try"])
    assert [a.review for a in tries] == [ReviewState.FAILED, ReviewState.PASSED] and made["s02"].id == tries[1].id
    assert tries[0].params["score"] is None and tries[0].params["checks"] == ["3 black frames"]
    assert lipsync.calls[0][4] != lipsync.calls[1][4]
    service.run(run.id, force=True)                             # forced: both tries are found by cache, none paid twice
    assert len(lipsync.calls) == 2


def test_a_clip_the_judge_fails_fails_the_stage_or_uses_its_best_try(stage, container):
    service, sheets, lipsync, ffmpeg, checks, vision = stage
    vision.fail_when, vision.fail_score = ("singing his line", 100), 4        # under the floor: no clip
    container.settings().image_max_tries = 1
    run = drawn_run(container, sheets)
    made = service.run(run.id)
    state = container.run_repository().stage(run.id, StageName.LIPSYNC)
    assert "s02" not in made and state.status == StageStatus.FAILED and "s02: failed every draw; best 4/10" in state.error
    vision.fail_score, container.settings().image_max_tries = 5, 2            # one more try, judged 5/10
    made = service.run(run.id)
    assert len(lipsync.calls) == 2 and made["s02"].review == ReviewState.USED_BEST and made["s02"].params["try"] == 2
    state = container.run_repository().stage(run.id, StageName.LIPSYNC)
    assert state.status == StageStatus.DONE and state.error is None and state.output["used_best"] == [made["s02"].id]


def test_the_cap_stops_a_run_before_drawing_and_keyframes_must_be_done(stage, container):
    service, sheets, lipsync, ffmpeg, checks, vision = stage
    run = drawn_run(container, sheets)
    container.settings().spend_cap_usd = container.asset_repository().run_cost(run.id) + 0.5
    with pytest.raises(ValueError, match="past the cap"):
        service.run(run.id)
    assert lipsync.calls == []
    assert container.run_repository().stage(run.id, StageName.LIPSYNC).status == StageStatus.FAILED
    with pytest.raises(ValueError, match="no finished keyframes"):
        service.run(ready_run(container, sheets).id)
