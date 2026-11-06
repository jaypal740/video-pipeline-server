import hashlib
import logging
import tempfile
from pathlib import Path

from src.core.configs import Settings
from src.repository.asset.asset_repository import AssetRepository
from src.repository.run.run_repository import RunRepository
from src.repository.song.song_repository import SongRepository
from src.schema.asset.asset_schema import Asset, ReviewState
from src.schema.base_schema import new_id, utcnow
from src.schema.enums import AssetKind, StageName, StageStatus
from src.schema.run.stage_state_schema import StageState
from src.schema.storyboard.storyboard_schema import Shot, ShotKind, Storyboard, Transition
from src.service.assembly.captions import caption_track
from src.service.local.ffmpeg_service import FfmpegService
from src.service.storage.storage_service import Storage

logger = logging.getLogger(__name__)

MAX_STRETCH = 1.25          # a model clip is slowed at most this much to fill its slot; then its last frame holds
BLACK_S = 0.5               # a black stretch longer than this fails the film
BLACK_PIXEL = 0.05          # darker than this is black ...
BLACK_PICTURE = 0.999       # ... and a frame is black only when nearly all of it is: a starry space sky is not
CAPTIONS_VERSION = "captions.v2"    # bumped when the captions change, so films are assembled again


class AssemblyService:
    """Stage 6: every shot's clip on the song's timeline, with the board's cuts and dissolves, the song laid
    on, one film stored and checked by code. Frames are counted from the song's start, so rounding never
    drifts. A model clip is retimed so all of it fills its slot (its last frame is the planned ending the
    next shot opens on); a lip-sync clip is never retimed, so the lips stay on the words."""

    def __init__(self, run_repository: RunRepository, asset_repository: AssetRepository,
                 song_repository: SongRepository, storage: Storage, ffmpeg: FfmpegService, settings: Settings) -> None:
        self._runs = run_repository
        self._assets = asset_repository
        self._songs = song_repository
        self._storage = storage
        self._ffmpeg = ffmpeg
        self._settings = settings

    def run(self, run_id: str, force: bool = False) -> Asset:
        board = self._board(run_id)
        clips = self._clips(run_id, board)
        s = self._settings
        fps, size, dissolve = s.assembly_fps, (s.assembly_width, s.assembly_height), s.assembly_dissolve_s
        song = self._songs.songs.get(self._runs.runs.get(run_id).song_id)
        analysis = self._songs.analyses.get(song.analysis_id) if song.analysis_id else None
        timing = [(w.line_index, w.text, w.start, w.end) for w in analysis.words] if analysis else []
        key = hashlib.sha256("|".join([f"{size[0]}x{size[1]}:{fps}:{dissolve}", f"{CAPTIONS_VERSION}:{timing}", *(
            f"{shot.id}:{shot.start}:{shot.end}:{shot.transition_in}:{clips[shot.id][0].id}" for shot in board.shots)])
                             .encode()).hexdigest()

        state = self._runs.stage(run_id, StageName.ASSEMBLY)
        if state is not None and not force and state.status == StageStatus.DONE and state.input_hash == key:
            logger.info("assembly %s: reused, inputs unchanged", run_id[:8])
            return self._assets.assets.get(state.output["film"])
        state = state or self._runs.stage_states.add(StageState(run_id=run_id, stage=StageName.ASSEMBLY))
        state.status, state.attempts, state.error = StageStatus.RUNNING, state.attempts + 1, None
        state.input_hash, state.started_at, state.finished_at = key, utcnow(), None
        state = self._runs.stage_states.update(state)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                work, rows = Path(tmp), []
                for i, shot in enumerate(board.shots):
                    asset, source = clips[shot.id]
                    raw = work / f"{shot.id}-raw.mp4"
                    raw.write_bytes(self._storage.get(asset.path))
                    frames = frames_of(board.shots, i, fps)
                    speed = self._fit(raw, frames, shot.kind != ShotKind.SUNG, work / f"{shot.id}.mp4",
                                      asset.params.get("active_seconds"))
                    rows.append({"shot": shot.id, "clip": asset.id, "source": source, "speed": speed,
                                 "frames": frames})
                film = work / "film.mp4"
                start = round(board.shots[0].start * fps) / fps
                total = sum(r["frames"] for r in rows)
                captions = (caption_track(analysis.lines, analysis.words, start, total / fps, *size, work)
                            if analysis and analysis.words else None)
                self._assemble(work, board.shots, rows, Path(song.audio_path), start, film, captions)
                problems = self._problems(film, total)
                data = film.read_bytes()
        except Exception as exc:
            state.status, state.error, state.finished_at = StageStatus.FAILED, str(exc), utcnow()
            self._runs.stage_states.update(state)
            logger.error("assembly %s: failed: %s", run_id[:8], exc)
            raise
        asset = self._store(run_id, data, total / fps, size, fps, problems)
        state.status = StageStatus.FAILED if problems else StageStatus.DONE
        state.error = "; ".join(problems) or None
        state.output = {"film": asset.id, "seconds": round(total / fps, 3), "shots": rows, "problems": problems}
        state.finished_at = utcnow()
        self._runs.stage_states.update(state)
        logger.log(logging.ERROR if problems else logging.INFO, "assembly %s: %s, %d shots, %.1fs, %s", run_id[:8],
                   state.status, len(rows), total / fps, "; ".join(problems) or "checks passed")
        return asset

    def _fit(self, raw: Path, frames: int, retime: bool, out: Path, active: float | None = None) -> float:
        """The clip at exactly `frames` frames, W x H, no audio. retime: the whole clip, first frame to last,
        spread over the slot (slowed at most MAX_STRETCH); otherwise frames from its start. A clip that holds
        still after `active` seconds (the video stage measured it) is cut there first, so the slot shows the
        action and ends on its last moving frame, not on a freeze. Either way a clip still short holds its
        last frame. Returns the playback speed (1.0 when not retimed)."""
        s = self._settings
        fps, w, h = s.assembly_fps, s.assembly_width, s.assembly_height
        stretch, trim = 1.0, ""
        if retime:
            source = (self._ffmpeg.packets(raw).get("video", 0) - 1) / self._ffmpeg.probe(raw)["fps"]
            if active and 0 < active < source:
                source, trim = active, f"trim=end={active:.3f},"
            stretch = min((frames - 1) / fps / source, MAX_STRETCH) if source > 0 else MAX_STRETCH
        self._ffmpeg.run(["-i", str(raw), "-an", "-vf",
                          f"{trim}setpts={stretch:.6f}*(PTS-STARTPTS),scale={w}:{h}:force_original_aspect_ratio=increase,"
                          f"crop={w}:{h},fps={fps},setsar=1,tpad=stop_mode=clone:stop_duration={frames / fps + 1},"
                          f"format=yuv420p", "-frames:v", str(frames), "-c:v", "libx264", "-crf", "18", str(out)])
        return round(1 / stretch, 4)

    def _assemble(self, work: Path, shots: list[Shot], rows: list[dict], song: Path, start: float, out: Path,
                  captions: Path | None = None) -> None:
        """Clips in board order. A dissolve overlaps the incoming clip by the dissolve length, held on its
        first frame, so every clip still starts at its own slot time and the song stays in sync. settb after
        every join keeps xfade's inputs on one timebase. captions: the lyric pictures laid over the whole film."""
        fps, d = self._settings.assembly_fps, self._settings.assembly_dissolve_s
        inputs, chain, length, label = [], [], 0.0, None
        for i, (shot, row) in enumerate(zip(shots, rows)):
            inputs += ["-i", str(work / f"{shot.id}.mp4")]
            if label is None:
                chain.append(f"[{i}:v]settb=1/{fps}[v{i}]")
            elif shot.transition_in == Transition.DISSOLVE:
                chain.append(f"[{i}:v]tpad=start_mode=clone:start_duration={d},settb=1/{fps}[p{i}]")
                chain.append(f"[{label}][p{i}]xfade=transition=fade:duration={d}:offset={length - d:.4f}"
                             f",settb=1/{fps}[v{i}]")
            else:
                chain.append(f"[{i}:v]settb=1/{fps}[c{i}]")
                chain.append(f"[{label}][c{i}]concat=n=2:v=1:a=0,settb=1/{fps}[v{i}]")
            label, length = f"v{i}", length + row["frames"] / fps
        if captions is not None:
            inputs += ["-f", "concat", "-safe", "0", "-i", str(captions)]
            chain.append(f"[{len(shots)}:v]fps={fps},format=rgba,settb=1/{fps}[cap]")
            chain.append(f"[{label}][cap]overlay=0:0:eof_action=pass,format=yuv420p[film]")
            label = "film"
        inputs += ["-ss", f"{start:.4f}", "-i", str(song)]
        self._ffmpeg.run([*inputs, "-filter_complex", ";".join(chain), "-map", f"[{label}]",
                          "-map", f"{len(shots) + (captions is not None)}:a",
                          "-t", f"{length:.4f}", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac",
                          "-b:a", "192k", "-movflags", "+faststart", str(out)])

    def _problems(self, film: Path, frames: int) -> list[str]:
        fps, found = self._settings.assembly_fps, []
        packets = self._ffmpeg.packets(film)
        if abs(packets.get("video", 0) - frames) > 1:
            found.append(f"the film is {packets.get('video', 0) / fps:.2f}s, the timeline is {frames / fps:.2f}s")
        if not packets.get("audio"):
            found.append("the film has no sound")
        for a, b in self._ffmpeg.black(film, BLACK_S, BLACK_PIXEL, BLACK_PICTURE):
            found.append(f"black from {a:.2f}s to {b:.2f}s")
        return found

    def _store(self, run_id: str, data: bytes, seconds: float, size: tuple[int, int], fps: int,
               problems: list[str]) -> Asset:
        asset_id = new_id()
        path = f"runs/{run_id}/film/{asset_id[:8]}.mp4"
        self._storage.put(path, data, "video/mp4")
        return self._assets.assets.add(Asset(
            id=asset_id, run_id=run_id, stage=StageName.ASSEMBLY, kind=AssetKind.VIDEO, path=path,
            sha256=hashlib.sha256(data).hexdigest(), provider="ffmpeg",
            params={"seconds": round(seconds, 3), "width": size[0], "height": size[1], "fps": fps, "checks": problems},
            cost_usd=0.0, review=ReviewState.FAILED if problems else ReviewState.PASSED,
            review_notes="\n".join(problems) or None))

    # --- inputs -------------------------------------------------------------------------------------------

    def _board(self, run_id: str) -> Storyboard:
        state = self._runs.stage(run_id, StageName.STORYBOARD)
        if state is None or state.status != StageStatus.DONE:
            raise ValueError(f"Run {run_id} has no storyboard; run the storyboard stage first")
        return Storyboard(**state.output)

    def _clips(self, run_id: str, board: Storyboard) -> dict[str, tuple[Asset, StageName]]:
        """Each shot's clip and the stage it came from: lip-sync for sung shots, video for the rest."""
        stages = [StageName.VIDEO] + ([StageName.LIPSYNC] if any(s.kind == ShotKind.SUNG for s in board.shots) else [])
        made = {}
        for name in stages:
            state = self._runs.stage(run_id, name)
            if state is None or state.status != StageStatus.DONE:
                raise ValueError(f"Run {run_id} has no finished {name} stage ({state.status if state else 'not run'});"
                                 f" run it first")
            made[name] = state.output.get("clips", {})
        clips, missing = {}, []
        for shot in board.shots:
            source = StageName.LIPSYNC if shot.kind == ShotKind.SUNG else StageName.VIDEO
            asset = self._assets.assets.get(made[source][shot.id]) if shot.id in made[source] else None
            if asset is None:
                missing.append(shot.id)
            else:
                clips[shot.id] = (asset, source)
        if missing:
            raise ValueError(f"Run {run_id} cannot be assembled: no clip for {', '.join(missing)}")
        return clips


def frames_of(shots: list[Shot], i: int, fps: int) -> int:
    """Frames in shot i's slot, to the next shot's start (the last to its end), counted from the song's start
    so rounding never adds up into drift."""
    end = shots[i + 1].start if i + 1 < len(shots) else shots[i].end
    return round(end * fps) - round(shots[i].start * fps)
