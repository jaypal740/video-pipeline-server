import hashlib
import io
import json
import logging
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from src.core.configs import Settings
from src.repository.asset.asset_repository import AssetRepository
from src.repository.run.run_repository import RunRepository
from src.repository.song.song_repository import SongRepository
from src.schema.asset.asset_schema import Asset, ReviewState
from src.schema.base_schema import new_id, utcnow
from src.schema.enums import AssetKind, StageName, StageStatus
from src.schema.run.stage_state_schema import StageState
from src.schema.storyboard.script_schema import DrawingKind
from src.service.local.ffmpeg_service import FfmpegService
from src.service.render.painter import Painter
from src.service.render.timeline import frame_at, scenes_of
from src.service.stage.stage_common import fail_stage, script_of
from src.service.storage.storage_service import Storage

logger = logging.getLogger(__name__)

STAGE = StageName.RENDER
VERSION = "render.v6"       # part of the stage's inputs: raise it when the same script should paint differently
USABLE = (ReviewState.PASSED, ReviewState.APPROVED, ReviewState.USED_BEST)
SPRITE_PX = 640             # a drawing's square is kept at this size; on screen it is drawn smaller than this


class RenderService:
    """Stage 6: the animation script played by code. Every frame is painted from the stored drawings and
    loops, where and when the script says, with the key words drawn on top, and the frames are joined into
    one film with the song under it. No model is called, so the same script gives the same film."""

    def __init__(self, run_repository: RunRepository, song_repository: SongRepository,
                 asset_repository: AssetRepository, storage: Storage, ffmpeg: FfmpegService, settings: Settings) -> None:
        self._runs = run_repository
        self._songs = song_repository
        self._assets = asset_repository
        self._storage = storage
        self._ffmpeg = ffmpeg
        self._settings = settings

    def run(self, run_id: str, until: float | None = None, force: bool = False) -> Asset:
        """until: render only the beats that start before that second of the song."""
        run = self._runs.runs.get(run_id)
        if run is None:
            raise LookupError(f"Run {run_id} not found")
        board = script_of(self._runs, run_id)
        beats = [b for b in board.beats if until is None or b.start < until]
        width, height, fps = self._settings.assembly_width, self._settings.assembly_height, self._settings.assembly_fps
        scenes = scenes_of(board, beats, width / height)
        moving = {d.id for d in board.drawings if d.motion}
        shown = sorted({t.drawing for s in scenes for t in s.tracks})
        drawn, looped = self._chosen(run_id, StageName.VISUAL_IDENTITY, "drawings"), self._chosen(run_id, StageName.LOOPS, "loops")
        backs, stills, loops = ({name: drawn.get(name) for name in sorted({s.background for s in scenes})},
                                {name: drawn.get(name) for name in shown if name not in moving},
                                {name: looped.get(name) for name in shown if name in moving})
        for found, stage in (({**backs, **stills}, "drawings"), (loops, "loops")):
            if missing := [name for name, asset in found.items() if asset is None]:
                raise ValueError(f"Run {run_id} has no usable {stage[:-1]} for {', '.join(missing)}; run the {stage} stage first")
        song = Path(self._songs.songs.get(run.song_id).audio_path)
        key = hashlib.sha256("|".join(
            [VERSION, f"{width}x{height}@{fps}", json.dumps([b.model_dump(mode="json") for b in beats], sort_keys=True)]
            + [f"{name}:{asset.sha256}" for name, asset in {**backs, **stills, **loops}.items()]).encode()).hexdigest()

        state = self._runs.stage(run_id, STAGE)
        if state is not None and not force and state.status == StageStatus.DONE and state.input_hash == key:
            logger.info("render %s: reused, inputs unchanged", run_id[:8])
            return self._assets.assets.get(state.output["film"])
        state = state or self._runs.stage_states.add(StageState(run_id=run_id, stage=STAGE))
        state.status, state.attempts, state.error = StageStatus.RUNNING, state.attempts + 1, None
        state.input_hash, state.started_at, state.finished_at = key, utcnow(), None
        state = self._runs.stage_states.update(state)
        seconds, frames = beats[-1].end, round(beats[-1].end * fps)
        logger.info("render %s: %d beats, %.1f s, %d frames at %dx%d, %d drawings and %d loops", run_id[:8], len(beats),
                    seconds, frames, width, height, len(stills), len(loops))
        try:
            with tempfile.TemporaryDirectory() as folder:
                painter = Painter(width, height, fps, {n: self._picture(a.path) for n, a in backs.items()},
                                  {n: self._picture(a.params["cut"]).resize((SPRITE_PX, SPRITE_PX), Image.LANCZOS)
                                   for n, a in stills.items()},
                                  {n: self._frames(Path(folder), n, a) for n, a in loops.items()})
                film = Path(folder, "film.mp4")
                self._ffmpeg.feed(
                    ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
                     "-i", str(song), "-map", "0:v", "-map", "1:a", "-t", f"{seconds:.3f}", "-c:v", "libx264", "-crf", "18",
                     "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(film)],
                    (painter.paint(frame_at(scenes, i / fps, width / height)).tobytes() for i in range(frames)))
                data = film.read_bytes()
        except Exception as exc:
            fail_stage(self._runs, state, exc, logger, "render")
            raise
        asset_id = new_id()
        path = f"runs/{run_id}/render/{asset_id[:8]}.mp4"
        self._storage.put(path, data, "video/mp4")
        asset = self._assets.assets.add(Asset(
            id=asset_id, run_id=run_id, stage=STAGE, kind=AssetKind.VIDEO, path=path, sha256=hashlib.sha256(data).hexdigest(),
            params={"seconds": seconds, "frames": frames, "width": width, "height": height, "fps": fps, "until": until},
            cost_usd=0.0))
        state.status, state.finished_at = StageStatus.DONE, utcnow()
        state.output = {"film": asset.id, "seconds": seconds, "until": until}
        self._runs.stage_states.update(state)
        logger.info("render %s: done, %s, %.1f MB, %.0fs", run_id[:8], path, len(data) / 1e6,
                    (state.finished_at - state.started_at).total_seconds())
        return asset

    def stills(self, run_id: str, times: list[float], height: int) -> list[Image.Image]:
        """Those seconds of the film painted from the drawings alone, `height` pixels tall, each drawing
        holding still where its loop will play: what the film will show, before any loop is paid for."""
        board = script_of(self._runs, run_id)
        wide = self._settings.assembly_width / self._settings.assembly_height
        scenes = scenes_of(board, board.beats, wide)
        drawn = self._chosen(run_id, StageName.VISUAL_IDENTITY, "drawings")
        shown = {s.background for s in scenes} | {t.drawing for s in scenes for t in s.tracks}
        if missing := sorted(shown - drawn.keys()):
            raise ValueError(f"Run {run_id} has no usable drawing for {', '.join(missing)}; run the drawings stage first")
        backs = {d.id for d in board.drawings if d.kind == DrawingKind.BACKGROUND}
        painter = Painter(round(height * wide), height, self._settings.assembly_fps,
                          {n: self._picture(drawn[n].path) for n in shown & backs},
                          {n: self._picture(drawn[n].params["cut"]) for n in shown - backs}, {})
        return [painter.paint(frame_at(scenes, t, wide)) for t in times]

    def _chosen(self, run_id: str, stage: StageName, what: str) -> dict[str, Asset]:
        """The usable drawings or loops a stage chose for the run."""
        state = self._runs.stage(run_id, stage)
        chosen = {name: self._assets.assets.get(a) for name, a in (state.output.get(what, {}) if state else {}).items()}
        return {name: asset for name, asset in chosen.items() if asset.review in USABLE}

    def _picture(self, key: str) -> Image.Image:
        return Image.open(io.BytesIO(self._storage.get(key))).convert("RGBA")

    def _frames(self, folder: Path, name: str, loop: Asset) -> np.ndarray:
        """A loop's see-through frames, read once into a file and mapped, so a film's loops need not all fit
        in memory."""
        clip, raw = folder / f"{name}.webm", folder / f"{name}.rgba"
        clip.write_bytes(self._storage.get(loop.params["cut"]))
        self._ffmpeg.run(["-c:v", "libvpx-vp9", "-i", str(clip), "-vf", f"scale={SPRITE_PX}:{SPRITE_PX}", "-f", "rawvideo",
                          "-pix_fmt", "rgba", str(raw)])
        return np.memmap(raw, dtype=np.uint8, mode="r").reshape(-1, SPRITE_PX, SPRITE_PX, 4)
