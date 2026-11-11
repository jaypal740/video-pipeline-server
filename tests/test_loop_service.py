import io
import subprocess

import numpy as np
import pytest
from PIL import Image
from dependency_injector import providers

from src.core.container import Container
from src.schema.asset.asset_schema import Asset, ReviewState
from src.schema.enums import AssetKind, StageName, StageStatus
from src.schema.lesson.lesson_schema import GradeBand
from src.schema.run.run_schema import Run
from src.schema.run.stage_state_schema import StageState
from src.schema.song.song_schema import Song
from src.service.drawing.cutout import KEY
from src.service.local.ffmpeg_service import FfmpegService
from src.service.loop.loop_checks import LoopCheck, LoopCutService, cut_frames, problems_of
from src.service.loop.loop_prompt import loop_brief, loop_prompt
from src.service.storage.storage_service import LocalStorage
from src.service.video.fal_video_service import VideoReply
from tests.test_drawing_service import BOARD, BY_ID, PLAN, SCRIPT, FakeVision, picture

FIX = "Its arms are held out wide."


def frames(path: list[int], size: int = 64, box: int = 24) -> np.ndarray:
    """A loop as the video model returns it: a green square on the key colour, its left edge at each x of `path`."""
    out = np.zeros((len(path), size, size, 3), dtype=np.uint8) + np.array(KEY, dtype=np.uint8)
    for i, x in enumerate(path):
        out[i, 20:20 + box, x:x + box] = (20, 160, 40)
    return out


def loop(**changes) -> LoopCheck:
    check = LoopCheck(seconds=4.0, fps=24.0, width=64, height=64, frames=96, gap=5.0, motion=150.0, uncut=[])
    check = check.model_copy(update=changes)
    return check.model_copy(update={"problems": problems_of(check)})


class FakeVideo:
    name = "fal"

    def __init__(self):
        self.calls = []

    def generate(self, model, prompt, image_url, seconds, seed=None, negative=None, end_image_url=None):
        self.calls.append((prompt, image_url, end_image_url, seconds))
        return VideoReply(data=f"clip {seed}".encode(), content_type="video/mp4", model=model, seed=seed,
                          seconds_requested=seconds, cost_usd=0.16, seconds=seconds, request_id="r")

    def upload(self, data, content_type):
        return f"https://fal.fake/{len(data)}"


class FakeCutter:
    """Reads every loop as good, except the checks queued in `found`, one per loop in the order made."""

    def __init__(self):
        self.found = []

    def cut(self, data):
        check = self.found.pop(0) if self.found else loop()
        return check, None if check.uncut else b"webm of " + data, picture((20, 160, 40))


@pytest.fixture
def stage(container: Container, tmp_path):
    video, cutter, vision, storage = FakeVideo(), FakeCutter(), FakeVision(), LocalStorage(str(tmp_path))
    container.storage.override(providers.Object(storage))
    container.fal_video_service.override(providers.Object(video))
    container.loop_cut_service.override(providers.Object(cutter))
    container.anthropic_service.override(providers.Object(vision))
    runs, assets = container.run_repository(), container.asset_repository()
    song = container.song_repository().songs.add(Song(name="colony", audio_path="a.mp3", lyrics="x"))
    run = runs.runs.add(Run(song_id=song.id, name="r", audience=GradeBand.G3_5))
    runs.stage_states.add(StageState(run_id=run.id, stage=StageName.LESSON, status=StageStatus.DONE,
                                     output=PLAN.model_dump(mode="json")))
    runs.stage_states.add(StageState(run_id=run.id, stage=StageName.STORYBOARD, status=StageStatus.DONE, output=BOARD))
    drawn = {}
    for d in SCRIPT.drawings:
        storage.put(f"drawings/{d.id}.png", picture((20, 160, 40)), "image/png")
        drawn[d.id] = assets.assets.add(Asset(run_id=run.id, stage=StageName.VISUAL_IDENTITY, kind=AssetKind.IMAGE,
                                              path=f"drawings/{d.id}.png", sha256=f"sha of {d.id}", target_id=d.id,
                                              review=ReviewState.PASSED)).id
    runs.stage_states.add(StageState(run_id=run.id, stage=StageName.VISUAL_IDENTITY, status=StageStatus.DONE,
                                     output={"drawings": drawn}))
    return container.loop_service(), video, cutter, vision, run


def test_a_loop_that_sways_and_comes_back_closes_and_moves_and_every_frame_is_cut_out():
    sink = io.BytesIO()
    gap, motion, furthest, uncut = cut_frames(frames([20, 26, 32, 26, 20]), sink)
    assert gap == 0.0 and motion > 100 and furthest == 2 and uncut == []
    cut = np.frombuffer(sink.getvalue(), dtype=np.uint8).reshape(5, 64, 64, 4)
    assert cut[2, 0, 0, 3] == 0 and cut[2, 30, 40, 3] == 255 and cut[0, 30, 50, 3] == 0      # the square, where it is


def test_a_still_loop_one_that_does_not_come_back_and_one_that_leaves_the_frame_are_problems():
    assert problems_of(loop()) == []
    assert "almost nothing moves" in problems_of(loop(motion=6.0))[0]
    assert "would jump each time it repeats" in problems_of(loop(gap=160.0))[0]
    gap, motion, _, uncut = cut_frames(frames([16, 24, 32, 36], box=32), io.BytesIO())    # slides to the edge and stays
    assert gap > 100 and len(uncut) == 2 and "the frame's edge" in uncut[0]
    assert problems_of(loop(uncut=uncut))[0].startswith("2 of 96 frames cannot be cut out: only")


def test_a_real_clip_is_read_cut_out_and_given_back_see_through(tmp_path):
    raw, clip = tmp_path / "loop.rgb", tmp_path / "loop.mp4"
    raw.write_bytes(frames([40, 52, 64, 52, 40] * 4, size=128, box=48).tobytes())
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "128x128", "-r", "10", "-i",
                    str(raw), "-pix_fmt", "yuv444p", str(clip)], check=True)
    check, webm, frame = LoopCutService(FfmpegService()).cut(clip.read_bytes())
    assert check.problems == [] and (check.frames, check.width, check.fps) == (20, 128, 10.0) and check.motion > 100
    assert Image.open(io.BytesIO(frame)).size == (128, 128)
    (tmp_path / "cut.webm").write_bytes(webm)
    back = subprocess.run(["ffmpeg", "-v", "error", "-c:v", "libvpx-vp9", "-i", str(tmp_path / "cut.webm"), "-frames:v", "1",
                           "-f", "rawvideo", "-pix_fmt", "rgba", "-"], capture_output=True, check=True).stdout
    first = np.frombuffer(back, dtype=np.uint8).reshape(128, 128, 4)
    assert first[2, 2, 3] == 0 and first[64, 64, 3] == 255                                 # clear around the square


def test_a_loop_is_asked_for_in_place_on_the_key_colour_and_judged_as_a_frame_of_the_drawing_moving():
    asked = loop_prompt(BY_ID["dot_point"])
    assert "It shows: Pointing up. Its motion: bobs." in asked and "magenta (#FF00FF)" in asked
    assert "ends exactly as it began" in asked
    brief = loop_brief(SCRIPT, BY_ID["dot_small"], ["One giant germ"])
    assert "wrong ideas: One giant germ." in brief.description and "in which it does this: wobbles." in brief.description
    assert brief.rules[-1].startswith("The subject is the same one as in the reference picture")
    assert brief.rules.count(brief.rules[-1]) == loop_brief(SCRIPT, BY_ID["dot_point"]).rules.count(brief.rules[-1]) == 1


def test_every_drawing_the_beats_show_moving_is_animated_once_checked_stored_and_costed(container, stage):
    service, video, _, vision, run = stage
    made = service.run(run.id)
    assert list(made) == ["dot_small", "dot_point"]            # dot has a motion but no beat shows it
    assert all(a.review == ReviewState.PASSED and a.kind == AssetKind.VIDEO for a in made.values())
    assert all(start == end and seconds == 4.0 for _, start, end, seconds in video.calls)   # back to the same drawing
    assert sorted(labels[1] for _, labels in vision.seen) == ["reference: the drawing dot_point",
                                                              "reference: the drawing dot_small"]
    storage, point = container.storage(), made["dot_point"]
    assert storage.get(point.path).startswith(b"clip ") and storage.get(point.params["cut"]).startswith(b"webm of clip ")
    assert point.params["cut"].endswith("-cut.webm") and point.params["motion"] == 150.0
    state = container.run_repository().stage(run.id, StageName.LOOPS)
    assert state.status == StageStatus.DONE and state.output["loops"]["dot_point"] == point.id
    assert container.asset_repository().run_cost(run.id) == pytest.approx(2 * 0.161)


def test_only_the_loops_the_first_seconds_need_are_made_and_the_same_drawings_reuse_their_loops(stage):
    service, video, _, _, run = stage
    assert list(service.run(run.id, until=4)) == ["dot_point"]
    first = service.run(run.id)
    assert service.run(run.id)["dot_small"].id == first["dot_small"].id and len(video.calls) == 2
    again = service.run(run.id, force=True)
    assert len(video.calls) == 2 and again["dot_small"].path == first["dot_small"].path and again["dot_small"].cost_usd == 0.0


def test_a_loop_that_fails_is_made_again_told_what_was_wrong(container, stage):
    service, video, cutter, vision, run = stage
    cutter.found = [loop(motion=6.0)]
    made = service.run(run.id, until=4)["dot_point"]
    assert made.review == ReviewState.PASSED and made.params["try"] == 2 and made.params["correction"] == FIX
    assert [prompt.endswith(f" {FIX}") for prompt, *_ in video.calls] == [False, True]
    assert "What the judge found wrong:\n- almost nothing moves" in vision.asked[0]


def test_when_the_second_loop_fails_too_the_better_one_is_used_and_the_stage_goes_on(container, stage):
    service, video, cutter, vision, run = stage
    cutter.found = [loop(), loop(gap=160.0)]                    # the first closes but the judge fails it; the second jumps
    vision.fail_first = 1
    made = service.run(run.id, until=4)["dot_point"]
    assert made.review == ReviewState.USED_BEST and made.params["try"] == 1 and made.params["score"] == 3
    state = container.run_repository().stage(run.id, StageName.LOOPS)
    assert state.status == StageStatus.DONE and state.output["used_best"] == ["dot_point"]
    assert service.run(run.id, until=4, force=True)["dot_point"].path == made.path and len(video.calls) == 2


def test_a_drawing_no_loop_of_which_can_be_cut_out_fails_the_stage(container, stage):
    service, video, cutter, vision, run = stage
    cutter.found = [loop(uncut=["only 60% of the frame's edge is flat"])] * 2
    made = service.run(run.id, until=4)["dot_point"]
    assert made.review == ReviewState.FAILED and made.params["cut"] is None and vision.seen == []
    state = container.run_repository().stage(run.id, StageName.LOOPS)
    assert state.status == StageStatus.FAILED and "dot_point: no loop could be cut out" in state.error


def test_a_run_needs_its_drawings_before_its_loops(container, stage):
    service, _, _, _, run = stage
    runs = container.run_repository()
    state = runs.stage(run.id, StageName.VISUAL_IDENTITY)
    state.output = {"drawings": {name: a for name, a in state.output["drawings"].items() if name != "dot_small"}}
    runs.stage_states.update(state)
    with pytest.raises(ValueError, match="no usable drawing for dot_small"):
        service.run(run.id)
    with pytest.raises(LookupError):
        service.run("missing")
