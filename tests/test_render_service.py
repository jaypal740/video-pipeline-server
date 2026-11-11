import io
import subprocess

import numpy as np
import pytest
from PIL import Image
from dependency_injector import providers

from src.core.configs import Settings
from src.core.container import Container
from src.schema.asset.asset_schema import Asset, ReviewState
from src.schema.enums import AssetKind, StageName, StageStatus
from src.schema.lesson.lesson_schema import GradeBand
from src.schema.run.run_schema import Run
from src.schema.run.stage_state_schema import StageState
from src.schema.song.song_schema import Song
from src.schema.storyboard.script_schema import Action, AnimationScript
from src.service.local.ffmpeg_service import FfmpegService
from src.service.render.painter import Painter
from src.service.render.timeline import CLOSE, EDGE, IN, ROW_FILL, ROW_Y, WIDE, Frame, Item, frame_at, scenes_of
from src.service.storage.storage_service import LocalStorage
from tests.test_drawing_service import BOARD, show
from tests.test_loop_service import frames

# The drawings stage's test script, with a third beat on another backdrop and more to do in it.
FILM = {**BOARD, "drawings": BOARD["drawings"] + [{"id": "lab", "kind": "background", "look": "A lab"}],
        "beats": BOARD["beats"] + [
            {"beat": "b03", "start": 8, "end": 14, "type": "explain", "background": "lab",
             "actions": [show("dot", 8), show("dish", 8, "left"), {"verb": "pull_back", "at": 8.5, "until": 9.5},
                         {"verb": "move", "drawing": "dish", "pos": "right", "at": 9, "until": 10},
                         {"verb": "zoom", "drawing": "dot", "scale": 1.5, "at": 10, "until": 11},
                         {"verb": "highlight", "drawing": "dot", "at": 11, "until": 12},
                         {"verb": "keyword", "text": "colony", "at": 11, "until": 13},
                         {"verb": "hide", "drawing": "dish", "at": 12},
                         {"verb": "lipsync", "drawing": "dot", "at": 8, "until": 14}]}]}
SCRIPT = AnimationScript(**FILM)
SHAPE = 16 / 9
SCENES = scenes_of(SCRIPT, SCRIPT.beats, SHAPE)


def on_screen(t: float) -> dict[str, Item]:
    return {i.drawing: i for i in frame_at(SCENES, t, SHAPE).items}


def png(colour, size=(64, 64), alpha=255) -> bytes:
    out = io.BytesIO()
    Image.new("RGBA", size, (*colour, alpha)).save(out, format="PNG")
    return out.getvalue()


def test_drawings_stay_through_beats_on_one_background_and_a_new_background_clears_the_screen():
    assert [(s.background, s.start, s.end) for s in SCENES] == [("bench", 0, 8), ("lab", 8, 14)]
    assert list(on_screen(0.5)) == ["dish"] and list(on_screen(3)) == ["dish", "dot_point"]     # the object under the performer
    assert list(on_screen(6.5)) == ["dish_full", "dot_point", "arrow"]                       # carried into b02, icon on top
    assert on_screen(7.9)["dot_small"].age == pytest.approx(0.9)                             # its loop began when it came
    assert list(on_screen(8.0)) == ["dish", "dot"] and frame_at(SCENES, 8.0, SHAPE).background == "lab"


def test_a_show_pops_in_a_swap_fades_one_into_the_other_and_a_hide_fades_out():
    coming = on_screen(0.1)["dish"]
    assert 0 < coming.opacity < 1 and coming.size < on_screen(1)["dish"].size == pytest.approx(0.44)
    assert on_screen(IN / 2)["dish"].size > 0.44 * 0.9                                       # most of the way out already
    both = on_screen(5.15)
    assert both["dish"].opacity + both["dish_full"].opacity == pytest.approx(1.0) and both["dish_full"].x == both["dish"].x
    assert "dish" not in on_screen(5.4) and on_screen(5.4)["dish_full"].opacity == 1
    assert 0 < on_screen(12.1)["dish"].opacity < 1 and "dish" not in on_screen(12.3)


def test_a_move_slides_a_zoom_grows_and_a_highlight_pulses_then_settles():
    assert on_screen(8.5)["dish"].x == pytest.approx(0.22) and on_screen(9.5)["dish"].x == pytest.approx(0.5)
    assert on_screen(10.5)["dish"].x == pytest.approx(0.78)
    assert on_screen(9.5)["dot"].size == pytest.approx(0.62) and on_screen(10.5)["dot"].size == pytest.approx(0.62 * 1.25)
    assert on_screen(11.25)["dot"].size == pytest.approx(0.62 * 1.5 * 1.08) and on_screen(12.5)["dot"].size == pytest.approx(0.93)


def test_the_camera_pushes_in_from_wide_and_a_scene_that_opens_on_a_pull_back_starts_close():
    assert frame_at(SCENES, 8.2, SHAPE).zoom == CLOSE and frame_at(SCENES, 9.0, SHAPE).zoom == pytest.approx((CLOSE + WIDE) / 2)
    assert frame_at(SCENES, 13, SHAPE).zoom == WIDE and frame_at(SCENES, 1, SHAPE).zoom == WIDE
    pushed = scenes_of(SCRIPT, [SCRIPT.beats[0].model_copy(update={"actions": SCRIPT.beats[0].actions + [
        SCRIPT.beats[2].actions[2].model_copy(update={"verb": "push_in", "at": 1, "until": 2}),
        SCRIPT.beats[2].actions[2].model_copy(update={"verb": "pan", "pos": "right", "at": 2, "until": 3})]})], SHAPE)
    assert frame_at(pushed, 0.5, SHAPE).zoom == WIDE and frame_at(pushed, 2.5, SHAPE).zoom == CLOSE
    assert frame_at(pushed, 2.5, SHAPE).pan == pytest.approx(0.0125) and frame_at(pushed, 3.5, SHAPE).pan == pytest.approx(0.025)


def staged(*shown: dict, word: bool = False) -> Frame:
    """The sixth second of a beat that shows those drawings and, when asked, the key word."""
    said = [SCRIPT.beats[2].actions[6].model_copy(update={"at": 5, "until": 7})] if word else []
    beat = SCRIPT.beats[1].model_copy(update={"actions": [*(Action(**s) for s in shown), *said]})
    return frame_at(scenes_of(SCRIPT, [beat], SHAPE), 6, SHAPE)


def test_a_key_word_takes_the_spot_that_covers_the_least_of_the_drawings_and_pops_in_and_out():
    assert staged(show("dot", 4), word=True).words == [("colony", 0.5, 0.13, 1.0)]          # clear above the performer
    assert staged(show("arrow", 4, "top"), show("dish", 4, "left"), word=True).words[0][1:3] == (0.78, 0.3)
    crowded = staged(show("dot", 4, "top"), show("dish", 4, "left"), show("dish_full", 4, "right"), word=True)
    assert crowded.words[0][1:3] == (0.5, 0.72)                                             # under the performer up top
    assert frame_at(SCENES, 11.1, SHAPE).words[0][3] == pytest.approx(0.4) and frame_at(SCENES, 13.5, SHAPE).words == []


def test_a_drawing_too_big_for_its_place_is_moved_to_stay_whole_in_the_wide_cameras_view():
    high, low = staged(show("dot", 4, "top"), show("dish", 4, "bottom")).items[::-1]
    assert high.drawing == "dot" and high.y - high.size / 2 == pytest.approx(EDGE)          # its head was out of the frame
    assert low.y + low.size / 2 == pytest.approx(1 - EDGE) and (high.x, low.x) == (0.5, 0.5)
    assert on_screen(1)["dish"].y == 0.54 and staged(show("arrow", 4, "top")).items[0].y == 0.2    # these fit as placed


def test_a_row_of_stages_shares_the_view_out_left_to_right_with_an_arrow_to_each_next_stage():
    row = staged(show("dish", 4, "row"), show("dot", 4.5, "row"), show("dot_point", 5, "row"))
    dish, dot, point = (next(i for i in row.items if i.drawing == d) for d in ("dish", "dot", "dot_point"))
    place = (1 - 2 * EDGE) / 3
    assert (dish.x, dot.x, point.x) == pytest.approx((EDGE + place / 2, 0.5, 1 - EDGE - place / 2))
    assert dish.y == dot.y == point.y == ROW_Y and dot.age == 1.5                           # each still plays its loop
    assert dot.size / SHAPE == pytest.approx(ROW_FILL * place) and dish.size == pytest.approx(0.44)   # made to fit its place
    assert [(round(x, 3), y, popped) for x, y, _, popped in row.arrows] == [
        (round(EDGE + place, 3), ROW_Y, 1.0), (round(EDGE + 2 * place, 3), ROW_Y, 1.0)]     # half way between two stages
    assert row.arrows[0][2] < (1 - ROW_FILL * 0.8) * place and row.words == []
    early = frame_at(scenes_of(SCRIPT, [SCRIPT.beats[1].model_copy(update={"actions": [
        Action(**show("dish", 4, "row")), Action(**show("dot", 7, "row"))]})], SHAPE), 6, SHAPE)
    assert [i.drawing for i in early.items] == ["dish"] and early.arrows == [] and early.items[0].x < 0.3   # its place is kept
    near = staged(show("dot_small", 4, "row"), show("dot", 4.5, "row"), show("dot_point", 5, "row"),
                  SCRIPT.beats[2].actions[2].model_dump() | {"verb": "push_in", "at": 5, "until": 5.5})
    edges = [0.5 + (i.x + side * i.size / SHAPE / 2 - 0.5) * near.zoom for i in near.items for side in (-1, 1)]
    assert near.zoom == CLOSE and 0 < min(edges) < 0.02 and 0.98 < max(edges) < 1          # the camera cuts no stage off


def test_a_frame_is_painted_from_the_background_the_cut_outs_and_the_loops_frame_for_its_age():
    red, blue = Image.new("RGBA", (40, 40), (220, 0, 0, 255)), np.zeros((4, 40, 40, 4), dtype=np.uint8)
    blue[..., 3] = 255
    blue[:, :, :, 2] = np.array([50, 100, 150, 200], dtype=np.uint8)[:, None, None]         # each frame its own blue
    painter = Painter(200, 100, 10, {"bench": Image.new("RGBA", (200, 100), (0, 120, 0, 255))}, {"dish": red}, {"dot": blue})

    def paint(zoom=1.0, pan=0.0, age=0.0, opacity=1.0, words=(), arrows=()):
        items = [Item(drawing="dish", x=0.25, y=0.5, size=0.4, opacity=opacity, age=0),
                 Item(drawing="dot", x=0.75, y=0.5, size=0.4, opacity=1, age=age)]
        return np.asarray(painter.paint(Frame(background="bench", zoom=zoom, pan=pan, items=items, words=list(words),
                                              arrows=list(arrows))))

    plain = paint()
    assert tuple(plain[5, 5]) == (0, 120, 0) and tuple(plain[50, 50]) == (220, 0, 0) and tuple(plain[50, 150]) == (0, 0, 50)
    assert tuple(plain[50, 75]) == (0, 120, 0)                                              # a 40 px square, 30 to 70
    assert tuple(paint(age=0.15)[50, 150]) == (0, 0, 100) and tuple(paint(age=0.3)[50, 150]) == (0, 0, 50)   # round again
    assert tuple(paint(zoom=2.0)[50, 75]) == (0, 120, 0) and tuple(paint(zoom=2.0)[50, 5]) == (220, 0, 0)    # from the middle
    assert tuple(paint(pan=0.1)[50, 25]) == (220, 0, 0) and tuple(paint(pan=0.1)[50, 55]) == (0, 120, 0)     # camera to the right
    assert tuple(paint(opacity=0.5)[50, 50]) == (110, 60, 0)
    word = paint(words=[("colony", 0.5, 0.5, 1.0)])
    assert (word[40:60, 70:130] == 255).all(axis=2).any() and tuple(word[5, 5]) == (0, 120, 0)              # a white plate
    arrow = paint(arrows=[(0.5, 0.15, 0.2, 1.0)])                                           # 40 px long, about (100, 15)
    assert tuple(arrow[15, 95]) == (255, 255, 255) and tuple(arrow[15, 75]) == tuple(arrow[3, 85]) == (0, 120, 0)
    long = paint(words=[("metamorphosis", 0.78, 0.3, 1.0)])                                 # smaller, and moved in to fit
    assert (long[30, 125:190] == 255).all(axis=1).any() and tuple(long[30, 108]) == tuple(long[30, 198]) == (0, 120, 0)


@pytest.fixture
def stage(container: Container, settings: Settings, tmp_path):
    """A run with its script, a picture for every drawing and a real two-frame loop for every moving one."""
    container.settings.override(providers.Object(settings.model_copy(update={
        "assembly_width": 160, "assembly_height": 90, "assembly_fps": 6})))
    storage = LocalStorage(str(tmp_path / "store"))
    container.storage.override(providers.Object(storage))
    container.ffmpeg_service.override(providers.Object(FfmpegService()))
    song = tmp_path / "song.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=20", str(song)], check=True)
    raw, loop = tmp_path / "loop.rgba", tmp_path / "loop.webm"
    raw.write_bytes(np.dstack([frames([20, 26])[0], np.full((64, 64), 255, np.uint8)]).tobytes() * 3)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", "64x64", "-r", "6", "-i", str(raw),
                    "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-auto-alt-ref", "0", str(loop)], check=True)
    runs, assets = container.run_repository(), container.asset_repository()
    record = container.song_repository().songs.add(Song(name="colony", audio_path=str(song), lyrics="x"))
    run = runs.runs.add(Run(song_id=record.id, name="r", audience=GradeBand.G3_5))
    runs.stage_states.add(StageState(run_id=run.id, stage=StageName.STORYBOARD, status=StageStatus.DONE, output=FILM))
    drawn, looped = {}, {}
    for d in SCRIPT.drawings:
        back = d.kind == "background"
        storage.put(f"d/{d.id}.png", png((0, 120, 0) if back else (220, 0, 0), (160, 90) if back else (64, 64)), "image/png")
        drawn[d.id] = assets.assets.add(Asset(
            run_id=run.id, stage=StageName.VISUAL_IDENTITY, kind=AssetKind.IMAGE, path=f"d/{d.id}.png", sha256=f"still {d.id}",
            target_id=d.id, params={"cut": None if back else f"d/{d.id}.png"}, review=ReviewState.PASSED)).id
        if d.motion:
            storage.put(f"l/{d.id}.webm", loop.read_bytes(), "video/webm")
            looped[d.id] = assets.assets.add(Asset(
                run_id=run.id, stage=StageName.LOOPS, kind=AssetKind.VIDEO, path=f"l/{d.id}.mp4", sha256=f"loop {d.id}",
                target_id=d.id, params={"cut": f"l/{d.id}.webm", "fps": 6.0}, review=ReviewState.PASSED)).id
    runs.stage_states.add(StageState(run_id=run.id, stage=StageName.VISUAL_IDENTITY, status=StageStatus.DONE,
                                     output={"drawings": drawn}))
    runs.stage_states.add(StageState(run_id=run.id, stage=StageName.LOOPS, status=StageStatus.DONE, output={"loops": looped}))
    return container.render_service(), run, tmp_path


def test_the_script_is_rendered_as_one_film_with_the_song_and_stored(container, stage):
    service, run, tmp_path = stage
    film = service.run(run.id)
    assert film.kind == AssetKind.VIDEO and film.params["frames"] == 84 and film.cost_usd == 0.0
    out = tmp_path / "film.mp4"
    out.write_bytes(container.storage().get(film.path))
    ffmpeg = FfmpegService()
    info = ffmpeg.probe(out)
    assert (info["width"], info["height"], info["fps"]) == (160, 90, 6.0) and info["seconds"] == pytest.approx(14, abs=0.3)
    assert ffmpeg.packets(out)["video"] == 84 and ffmpeg.packets(out)["audio"] > 0
    frame = np.frombuffer(ffmpeg.run(["-ss", "3", "-i", str(out), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]),
                          dtype=np.uint8).reshape(90, 160, 3)
    assert frame[5, 5, 1] > 90 and frame[5, 5, 0] < 40 and frame[49, 80, 0] > 180           # the backdrop, and the dish on it
    state = container.run_repository().stage(run.id, StageName.RENDER)
    assert state.status == StageStatus.DONE and state.output == {"film": film.id, "seconds": 14.0, "until": None}
    assert service.run(run.id).id == film.id and service.run(run.id, force=True).id != film.id


def test_only_the_beats_that_start_before_a_second_are_rendered(stage):
    service, run, _ = stage
    assert service.run(run.id, until=3).params["seconds"] == 4 and service.run(run.id, until=5).params["seconds"] == 8


def test_a_render_needs_the_drawings_and_loops_its_beats_show(container, stage):
    service, run, _ = stage
    runs = container.run_repository()
    loops = runs.stage(run.id, StageName.LOOPS)
    loops.output = {"loops": {name: a for name, a in loops.output["loops"].items() if name != "dot_small"}}
    runs.stage_states.update(loops)
    with pytest.raises(ValueError, match="no usable loop for dot_small; run the loops stage first"):
        service.run(run.id)
    assert service.run(run.id, until=3).params["seconds"] == 4          # the first beat does not show it
    drawn = runs.stage(run.id, StageName.VISUAL_IDENTITY)
    drawn.output = {"drawings": {name: a for name, a in drawn.output["drawings"].items() if name != "bench"}}
    runs.stage_states.update(drawn)
    with pytest.raises(ValueError, match="no usable drawing for bench; run the drawings stage first"):
        service.run(run.id, until=3, force=True)
    with pytest.raises(LookupError):
        service.run("missing")
