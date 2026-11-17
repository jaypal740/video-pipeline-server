import hashlib
import subprocess

import numpy as np
import pytest

from src.core.container import Container
from src.schema.asset.asset_schema import Asset, ReviewState
from src.schema.enums import AssetKind, StageName, StageStatus
from src.schema.lesson.lesson_schema import GradeBand
from src.schema.run.run_schema import Run
from src.schema.run.stage_state_schema import StageState
from src.schema.song.song_analysis_schema import LyricLine, SongAnalysis, TimedWord
from src.schema.song.song_schema import Song
from src.service.assembly.assembly_service import AssemblyService
from src.service.storage.storage_service import LocalStorage
from tests.test_keyframe_service import shot
from tests.test_sheet_service import BOARD

FPS = 24
RED, BLUE, YELLOW, CYAN = (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)


def ff(*args) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", *args], check=True)


def clip(path, *parts) -> bytes:
    """A 160x90 clip at FPS of solid colours in turn: parts are (colour name, seconds)."""
    inputs = [a for name, s in parts for a in ("-f", "lavfi", "-i", f"color={name}:size=160x90:rate={FPS}:d={s}")]
    joined = "".join(f"[{i}:v]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=1:a=0[v]"
    ff(*inputs, "-filter_complex", joined, "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path))
    return path.read_bytes()


def colour_at(film, seconds) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{seconds:.4f}", "-i", str(film), "-frames:v", "1", "-f",
                          "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).mean(axis=0)


def near(found, wanted) -> bool:
    return bool(np.all(np.abs(found - np.array(wanted)) < 40))


@pytest.fixture
def stage(container: Container, tmp_path):
    settings = container.settings()
    settings.assembly_width, settings.assembly_height = 160, 90
    storage = LocalStorage(str(tmp_path / "files"))
    service = AssemblyService(container.run_repository(), container.asset_repository(), container.song_repository(),
                              storage, container.ffmpeg_service(), settings)
    song = tmp_path / "song.wav"
    ff("-f", "lavfi", "-i", "sine=frequency=440:duration=5", str(song))
    return service, storage, song, tmp_path


def assembled_run(container, storage, song, tmp_path, clips=None):
    """s01 0-1.5 model clip 2.0 s (red, ends blue); s02 1.5-2.5 sung lip-sync clip 2.0 s (yellow, then green);
    s03 2.5-4.0 camera clip 1.5 s (cyan), dissolving in."""
    shots = [{**shot("s01", "illustrate", "sea", ["ship"], "wide"), "start": 0.0, "end": 1.5},
             {**shot("s02", "sung", "sea", ["host"], "medium_close"), "start": 1.5, "end": 2.5},
             {**shot("s03", "illustrate", "sea", ["ship"], "wide"), "start": 2.5, "end": 4.0,
              "transition_in": "dissolve"}]
    clips = clips or {"s01": [("red", 1.7), ("blue", 0.3)], "s02": [("yellow", 1.0), ("green", 1.0)],
                      "s03": [("cyan", 1.5)]}
    runs, assets = container.run_repository(), container.asset_repository().assets
    record = container.song_repository().songs.add(Song(name=f"song-{len(runs.runs.list())}", audio_path=str(song),
                                                        lyrics="x"))
    run = runs.runs.add(Run(song_id=record.id, name="r", audience=GradeBand.G3_5))
    made = {}
    for shot_id, parts in clips.items():
        path = f"runs/{run.id}/clips/{shot_id}.mp4"
        data = clip(tmp_path / f"{run.id[:8]}-{shot_id}.mp4", *parts)
        storage.put(path, data, "video/mp4")
        stage = StageName.LIPSYNC if shot_id == "s02" else StageName.VIDEO
        made[shot_id] = assets.add(Asset(run_id=run.id, stage=stage, kind=AssetKind.VIDEO, path=path,
                                         sha256=hashlib.sha256(data).hexdigest(), target_id=shot_id,
                                         params={"route": "camera" if shot_id == "s03" else "model"},
                                         review=ReviewState.PASSED))
    runs.stage_states.add(StageState(run_id=run.id, stage=StageName.STORYBOARD, status=StageStatus.DONE,
                                     output={**BOARD, "shots": shots}))
    runs.stage_states.add(StageState(run_id=run.id, stage=StageName.VIDEO, status=StageStatus.DONE,
                                     output={"clips": {s: a.id for s, a in made.items() if s != "s02"}}))
    runs.stage_states.add(StageState(run_id=run.id, stage=StageName.LIPSYNC, status=StageStatus.DONE,
                                     output={"clips": {s: a.id for s, a in made.items() if s == "s02"}}))
    return run, made


def test_clips_are_fitted_to_their_slots_on_the_song_timeline(stage, container):
    service, storage, song, tmp_path = stage
    run, made = assembled_run(container, storage, song, tmp_path)
    film = service.run(run.id)
    state = container.run_repository().stage(run.id, StageName.ASSEMBLY)
    assert state.status == StageStatus.DONE and state.output["problems"] == [] and film.review == ReviewState.PASSED
    assert film.stage == StageName.ASSEMBLY and film.path == f"runs/{run.id}/film/{film.id[:8]}.mp4"
    rows = {r["shot"]: r for r in state.output["shots"]}
    assert [rows[s]["frames"] for s in ("s01", "s02", "s03")] == [36, 24, 36] and state.output["seconds"] == 4.0
    assert rows["s01"]["speed"] > 1.0 and rows["s01"]["source"] == StageName.VIDEO   # 2.0 s sped up into 1.5 s
    assert rows["s02"] == {"shot": "s02", "clip": made["s02"].id, "source": StageName.LIPSYNC, "speed": 1.0,
                           "frames": 24}
    path = storage.path(film.path)
    packets = container.ffmpeg_service().packets(path)
    assert packets["video"] == 96 and packets["audio"] > 0                 # the song timeline, to the frame
    assert near(colour_at(path, 0.5), RED)
    assert near(colour_at(path, 1.5 - 1 / FPS - 0.01), BLUE)          # the whole clip: its ending is shown
    assert near(colour_at(path, 1.5 + 0.1), YELLOW) and near(colour_at(path, 2.08), YELLOW)  # trimmed
    assert near(colour_at(path, 2.5 - 0.2), (128, 255, 128))              # mid-dissolve: yellow into cyan
    assert near(colour_at(path, 2.5 + 0.02), CYAN) and near(colour_at(path, 3.9), CYAN)   # s03 on its slot time


def test_reuse_when_nothing_changed_and_force_draws_again(stage, container):
    service, storage, song, tmp_path = stage
    run, _ = assembled_run(container, storage, song, tmp_path)
    film = service.run(run.id)
    assert service.run(run.id).id == film.id
    assert service.run(run.id, force=True).id != film.id


def test_a_missing_clip_fails_with_the_shot_named(stage, container):
    service, storage, song, tmp_path = stage
    run, _ = assembled_run(container, storage, song, tmp_path)
    runs = container.run_repository()
    video = runs.stage(run.id, StageName.VIDEO)
    video.output = {"clips": {}}
    runs.stage_states.update(video)
    with pytest.raises(ValueError, match="no clip for s01, s03"):
        service.run(run.id)


def test_a_black_stretch_fails_the_film_but_it_is_still_stored(stage, container):
    service, storage, song, tmp_path = stage
    run, _ = assembled_run(container, storage, song, tmp_path,
                           clips={"s01": [("black", 1.5)], "s02": [("yellow", 0.5)], "s03": [("cyan", 1.0)]})
    film = service.run(run.id)
    state = container.run_repository().stage(run.id, StageName.ASSEMBLY)
    assert state.status == StageStatus.FAILED and film.review == ReviewState.FAILED
    assert state.output["problems"][0].startswith("black from 0.00s") and storage.exists(film.path)
    s02, s03 = state.output["shots"][1:]
    assert s02["frames"] == 24 and s02["speed"] == 1.0                    # a short sung clip holds its last frame
    assert s03["frames"] == 36 and s03["speed"] == 0.8                    # slowed at most 1.25x, then held
    assert near(colour_at(storage.path(film.path), 3.95), CYAN)


def test_a_clip_that_holds_still_at_the_end_is_cut_where_it_stopped_moving(stage, container):
    service, storage, song, tmp_path = stage
    run, made = assembled_run(container, storage, song, tmp_path,
                              clips={"s01": [("red", 1.2), ("blue", 0.8)], "s02": [("yellow", 1.0)],
                                     "s03": [("cyan", 1.5)]})
    s01 = made["s01"]
    s01.params = {**s01.params, "active_seconds": 1.25}            # the video stage saw it hold blue from 1.25 s
    container.asset_repository().assets.update(s01)
    film = storage.path(service.run(run.id).path)
    rows = {r["shot"]: r for r in container.run_repository().stage(run.id, StageName.ASSEMBLY).output["shots"]}
    assert rows["s01"]["speed"] < 1.0                               # 1.25 s of action spread over the 1.5 s slot
    assert near(colour_at(film, 1.0), RED)                          # whole 2 s squeezed in would be blue by now
    assert near(colour_at(film, 1.5 - 1 / FPS - 0.01), BLUE)        # and it still ends on its ending


def test_the_lyrics_are_laid_over_the_film_while_each_line_is_sung(stage, container):
    service, storage, song, tmp_path = stage
    container.settings().assembly_width, container.settings().assembly_height = 640, 360
    run, _ = assembled_run(container, storage, song, tmp_path)
    songs = container.song_repository()
    analysis = songs.analyses.add(SongAnalysis(
        cache_key="k", audio_sha256="a", lyrics_sha256="l", duration_s=5, accepted=True,
        lines=[LyricLine(index=0, text="Red ship", start=0.2, end=1.2, confidence=0.9)],
        words=[TimedWord(index=0, line_index=0, text="Red", start=0.2, end=0.6),
               TimedWord(index=1, line_index=0, text="ship", start=0.7, end=1.2)]))
    record = songs.songs.get(run.song_id)
    record.analysis_id = analysis.id
    songs.songs.update(record)
    film = storage.path(service.run(run.id).path)
    assert container.ffmpeg_service().packets(film)["video"] == 96
    caption = lambda t: colour_at_box(film, t, 640, 360)
    assert near(caption(0.1), RED) and near(caption(1.4), BLUE)          # before and after the line: the shot
    assert near(caption(0.5), (102, 0, 0))                               # while it is sung: the box, 60% black


def colour_at_box(film, seconds, width, height) -> np.ndarray:
    """The mean colour of a small patch just inside the caption box's bottom edge, at the frame's centre."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{seconds:.4f}", "-i", str(film), "-frames:v", "1", "-f",
                          "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
    y = int(height - height * 0.08)
    return frame[y - 2:y, width // 2 - 5:width // 2 + 5].reshape(-1, 3).mean(axis=0)


def test_a_labelled_shot_gets_its_word_drawn_in_the_open_third(stage, container):
    service, storage, song, tmp_path = stage
    container.settings().assembly_width, container.settings().assembly_height = 640, 360
    run, _ = assembled_run(container, storage, song, tmp_path)
    runs = container.run_repository()
    board = runs.stage(run.id, StageName.STORYBOARD)
    board.output["shots"][0].update(label="ship", position="left_third")
    runs.stage_states.update(board)
    film = storage.path(service.run(run.id).path)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", "0.7", "-i", str(film), "-frames:v", "1", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    frame = np.frombuffer(raw, dtype=np.uint8).reshape(360, 640, 3)
    right, left = frame[120:180, 440:620], frame[120:180, 20:200]
    assert (right.min(axis=2) > 200).any() and not (left.min(axis=2) > 200).any()   # white word on the right only


def test_a_songs_singer_is_cut_out_of_her_green_screen_and_laid_over_the_real_clip(stage, container):
    service, storage, song, tmp_path = stage
    girl = {"id": "singer", "kind": "person", "look": "A girl in a green jacket", "size": "child", "scale": "human",
            "star": True}
    shots = [{**shot("s01", "illustrate", "sea", ["ship"], "wide"), "start": 0.0, "end": 1.5},
             {**shot("s02", "sung", "sea", ["singer", "ship"], "medium_close"), "start": 1.5, "end": 3.0,
              "performer": "singer"}]
    runs, assets = container.run_repository(), container.asset_repository().assets
    record = container.song_repository().songs.add(Song(name="singer-film", audio_path=str(song), lyrics="x"))
    run = runs.runs.add(Run(song_id=record.id, name="r", audience=GradeBand.G3_5))
    green = tmp_path / "green.mp4"                  # her clip: a red figure on a green screen that turns teal as she sings
    ff("-f", "lavfi", "-i", f"color=0x00b140:size=160x90:rate={FPS}:d=3", "-f", "lavfi", "-i",
       f"color=0x64adad:size=160x90:rate={FPS}:d=3", "-filter_complex",
       "[0][1]blend=all_expr='if(lt(T,1.2),A,B)',drawbox=x=60:y=20:w=40:h=70:color=red:t=fill",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", str(green))
    files = {"s01": (StageName.VIDEO, clip(tmp_path / "a.mp4", ("blue", 1.5))),
             "s02": (StageName.VIDEO, clip(tmp_path / "b.mp4", ("yellow", 1.5))),
             "her": (StageName.LIPSYNC, green.read_bytes())}
    made = {}
    for name, (stage_name, data) in files.items():
        storage.put(f"runs/{run.id}/{name}.mp4", data, "video/mp4")
        made[name] = assets.add(Asset(run_id=run.id, stage=stage_name, kind=AssetKind.VIDEO, path=f"runs/{run.id}/{name}.mp4",
                                      sha256=hashlib.sha256(data).hexdigest(), target_id=name, params={},
                                      review=ReviewState.PASSED))
    cast = [m for m in BOARD["cast"] if m["id"] == "ship"] + [girl]
    runs.stage_states.add(StageState(run_id=run.id, stage=StageName.STORYBOARD, status=StageStatus.DONE,
                                     output={**BOARD, "cast": cast, "shots": shots}))
    runs.stage_states.add(StageState(run_id=run.id, stage=StageName.VIDEO, status=StageStatus.DONE,
                                     output={"clips": {"s01": made["s01"].id, "s02": made["s02"].id}}))
    runs.stage_states.add(StageState(run_id=run.id, stage=StageName.LIPSYNC, status=StageStatus.DONE,
                                     output={"clips": {"s02": made["her"].id}, "offsets": {"s02": 0.5}}))
    film = service.run(run.id)
    state = runs.stage(run.id, StageName.ASSEMBLY)
    assert state.status == StageStatus.DONE and state.output["shots"][1]["source"] == StageName.VIDEO
    path = storage.path(film.path)
    for at in (1.8, 2.5):                           # the real clip, with her red figure on it: no green, and no teal after
        r, g, b = colour_at(path, at)
        assert r > 240 and 200 < g < 250 and b < 30
    assert near(colour_at(path, 1.5 - 0.2), (128, 128, 128))       # the look changes on a dissolve: blue into yellow
