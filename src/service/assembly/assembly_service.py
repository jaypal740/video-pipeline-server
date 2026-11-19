import hashlib
import logging
import math
import re
import tempfile
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter

from src.core.configs import Settings
from src.repository.asset.asset_repository import AssetRepository
from src.repository.run.run_repository import RunRepository
from src.repository.song.song_repository import SongRepository
from src.schema.asset.asset_schema import Asset, ReviewState
from src.schema.base_schema import new_id, utcnow
from src.schema.enums import AssetKind, StageName, StageStatus
from src.schema.lesson.lesson_schema import LessonPlan
from src.schema.song.song_analysis_schema import TimedWord
from src.schema.storyboard.storyboard_schema import Position, Shot, ShotKind, Storyboard, Transition
from src.service.assembly.captions import KEYWORD_SECONDS, Pop, Screen, caption_track, sung_at
from src.service.local.ffmpeg_service import FfmpegService
from src.service.stage.stage_common import start_stage, storyboard_of
from src.service.storage.storage_service import Storage
from src.service.storyboard.storyboard_rules import real_shots

logger = logging.getLogger(__name__)

MAX_STRETCH = 1.25          # a model clip is slowed at most this much to fill its slot; then its last frame holds
BLACK_S = 0.5               # a black stretch longer than this fails the film
BLACK_PIXEL = 0.05          # darker than this is black ...
BLACK_PICTURE = 0.999       # ... and a frame is black only when nearly all of it is: a starry space sky is not
CAPTIONS_VERSION = "captions.v3"    # bumped when the captions change, so films are assembled again
_NOT_LETTER = re.compile(r"[^a-z]")
# A song's singer is cut out of her green screen and stood at the left of the real footage she sings over.
SINGER_HEIGHT = 0.65        # her clip's height as a share of the film's
SINGER_CENTRE = 0.17        # where she stands across the film ...
SINGER_MARGIN = 0.01        # ... moved right as far as it takes to keep this much between her and its left edge
GREEN_NEAR, GREEN_BLEND = 0.13, 0.04    # how close to the screen's colour is removed, and how soft the edge is
GREEN_EXACT = 0.025         # ... and how close counts as the screen itself, wherever it is in her clip
SCREEN_EVERY = 0.5          # the length of each piece she is cut out in, in seconds


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
        real = real_shots(board.cast, board.shots)                  # real footage with the singer laid over it
        board.shots = [s.model_copy(update={"transition_in": Transition.DISSOLVE})      # the look changes softly
                       if i and (s.id in real) != (board.shots[i - 1].id in real) else s
                       for i, s in enumerate(board.shots)]
        clips = self._clips(run_id, board, real)
        singers = self._singers(run_id, real)
        s = self._settings
        fps, size, dissolve = s.assembly_fps, (s.assembly_width, s.assembly_height), s.assembly_dissolve_s
        song = self._songs.songs.get(self._runs.runs.get(run_id).song_id)
        analysis = self._songs.analyses.get(song.analysis_id) if song.analysis_id else None
        timing = [(w.line_index, w.text, w.start, w.end) for w in analysis.words] if analysis else []
        pops = keyword_pops(self._lesson(run_id), analysis.words if analysis else [], board.shots)
        screen = screen_words(board.shots, labels=not pops)
        key = hashlib.sha256("|".join([f"{size[0]}x{size[1]}:{fps}:{dissolve}",
                                       f"{CAPTIONS_VERSION}:{timing}:{screen}{pops or ''}", *(
            f"{shot.id}:{shot.start}:{shot.end}:{shot.transition_in}:{clips[shot.id][0].id}"
            f"{':' + singers[shot.id][0].id if shot.id in singers else ''}" for shot in board.shots)])
                             .encode()).hexdigest()

        state = self._runs.stage(run_id, StageName.ASSEMBLY)
        if state is not None and not force and state.status == StageStatus.DONE and state.input_hash == key:
            logger.info("assembly %s: reused, inputs unchanged", run_id[:8])
            return self._assets.assets.get(state.output["film"])
        state = start_stage(self._runs, state, run_id, StageName.ASSEMBLY, key)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                work, rows = Path(tmp), []
                for i, shot in enumerate(board.shots):
                    asset, source = clips[shot.id]
                    raw = work / f"{shot.id}-raw.mp4"
                    raw.write_bytes(self._storage.get(asset.path))
                    frames = frames_of(board.shots, i, fps)
                    speed = self._fit(raw, frames, source == StageName.VIDEO, work / f"{shot.id}.mp4",
                                      asset.params.get("active_seconds"))
                    if shot.id in singers:
                        singer, offset = singers[shot.id]
                        green = work / f"{shot.id}-singer.mp4"
                        green.write_bytes(self._storage.get(singer.path))
                        self._lay_singer(work / f"{shot.id}.mp4", green, offset, frames)
                    rows.append({"shot": shot.id, "clip": asset.id, "source": source, "speed": speed,
                                 "frames": frames})
                film = work / "film.mp4"
                start = round(board.shots[0].start * fps) / fps
                total = sum(r["frames"] for r in rows)
                lines, words = (analysis.lines, analysis.words) if analysis else ([], [])
                captions = (caption_track(lines, words, start, total / fps, *size, work, screen, pops)
                            if words or screen else None)
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
        if fps <= 0 or frames <= 0:
            raise ValueError(f"cannot fit a clip to {frames} frames at {fps} fps; both must be above zero")
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

    def _lay_singer(self, film: Path, singer: Path, offset: float, frames: int) -> None:
        """The singer, cut out of her green screen from `offset` seconds into her clip, stood at the bottom
        left of a fitted shot. The video model lets her screen drift to another colour as she sings, so she is
        cut out in pieces of SCREEN_EVERY seconds, each removing only the colour its screen has then, read from
        her clip's own corner. A near match is removed only where it reaches the frame's edge, so a patch of
        her skin or clothes close to the screen's colour stays; an exact match is removed anywhere, which
        clears the screen shut in between her arm and her side. Colours are matched in full, not by tint."""
        fps, h = self._settings.assembly_fps, self._settings.assembly_height
        seen = self._ffmpeg.probe(singer)
        w, sh, last = seen["width"], seen["height"], seen["seconds"] - 0.2
        edge = [(min(w - 3, 2 + i * w // 24), y) for i in range(25) for y in (2, sh - 3)] + \
               [(x, 2 + j * (sh - 5) // 8) for j in range(9) for x in (2, w - 3)]
        outside = ",".join(f"floodfill=x={x}:y={y}:s0=0:d0=100" for x, y in edge)
        corner, pieces = film.with_name(f"{film.stem}-corner.png"), []
        left, middle = self._stands(singer, corner)
        steps = max(1, math.ceil(frames / fps / SCREEN_EVERY))
        for i in range(steps):
            f0, f1 = round(frames * i / steps), round(frames * (i + 1) / steps)
            at = offset + f0 / fps
            colours = sorted({self._screen(singer, min(t, last), corner) for t in (at, offset + (f1 - 1) / fps)})
            keys = "".join(f"[k{j}]colorkey=0x{c}:{GREEN_NEAR}:{GREEN_BLEND},alphaextract[n{j}];"
                           f"[x{j}]colorkey=0x{c}:{GREEN_EXACT}:0,alphaextract[e{j}];" for j, c in enumerate(colours))
            near, exact = "n0", "e0"
            if len(colours) == 2:                               # the screen changed within the piece: remove both
                keys += "[n0][n1]blend=all_mode=darken[n];[e0][e1]blend=all_mode=darken[e];"
                near, exact = "n", "e"
            pieces.append(film.with_name(f"{film.stem}-sung{i:03d}.mp4"))
            self._ffmpeg.run(["-i", str(film), "-ss", f"{min(at, last):.3f}", "-i", str(singer), "-filter_complex",
                              f"[0:v]trim=start_frame={f0}:end_frame={f1},setpts=PTS-STARTPTS[f];"
                              f"[1:v]fps={fps},setpts=PTS-STARTPTS,format=rgba,split={2 * len(colours) + 1}[c]"
                              f"{''.join(f'[k{j}][x{j}]' for j in range(len(colours)))};{keys}"
                              f"[{near}]format=gray,lut=y='if(lt(val,128),0,255)',{outside},"
                              f"lut=y='if(eq(val,100),0,255)'[in];"
                              # her outline keeps a line of the screen's colour: trim the edge in by two pixels
                              f"[in][{exact}]blend=all_mode=darken,erosion,erosion,gblur=sigma=0.8[m];[c][m]alphamerge,"
                              f"scale=-2:{2 * round(h * SINGER_HEIGHT / 2)},"
                              f"tpad=stop_mode=clone:stop_duration={frames / fps + 1}[s];"
                              f"[f][s]overlay=x='max(main_w*{SINGER_CENTRE}-overlay_w*{middle:.3f},main_w*{SINGER_MARGIN}"
                              f"-overlay_w*{left:.3f})':y=main_h-overlay_h,format=yuv420p",
                              "-an", "-frames:v", str(f1 - f0), "-c:v", "libx264", "-crf", "18", str(pieces[-1])])
        joined = film.with_name(f"{film.stem}-sung.txt")
        joined.write_text("".join(f"file '{piece}'\n" for piece in pieces))
        out = film.with_name(f"{film.stem}-sung.mp4")
        self._ffmpeg.run(["-f", "concat", "-safe", "0", "-i", str(joined), "-c", "copy", str(out)])
        out.replace(film)

    def _stands(self, singer: Path, png: Path) -> tuple[float, float]:
        """Where she stands in her own clip, as shares of its width: the furthest left she reaches, and her
        middle. It is read from the whole clip, so she stays in one place through every shot of a chorus. She
        is not always in the middle of her clip, and placed as if she were, she was cut off by the film's edge."""
        self._ffmpeg.run(["-y", "-i", str(singer), "-vf", "fps=5,scale=104:60,tile=1x100", "-frames:v", "1", str(png)])
        sheet, lefts, middles = Image.open(png).convert("RGB"), [], []
        for top in range(0, sheet.height, 60):
            frame = sheet.crop((0, top, 104, top + 60))
            apart = ImageChops.difference(frame, Image.new("RGB", frame.size, frame.getpixel((1, 1)))).convert("L")
            box = apart.point(lambda v: 255 if v > 40 else 0).filter(ImageFilter.MinFilter(3)).getbbox()
            if box is not None:
                lefts.append(box[0] / 104)
                middles.append((box[0] + box[2]) / 208)
        return (min(lefts), sorted(middles)[len(middles) // 2]) if lefts else (0.0, 0.5)

    def _screen(self, singer: Path, at: float, png: Path) -> str:
        """The colour of the singer's screen `at` seconds into her clip, read from its corner."""
        self._ffmpeg.run(["-y", "-ss", f"{max(at, 0):.3f}", "-i", str(singer), "-frames:v", "1", str(png)])
        return "%02x%02x%02x" % Image.open(png).convert("RGB").getpixel((8, 8))

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
        return storyboard_of(self._runs, run_id)

    def _lesson(self, run_id: str) -> LessonPlan | None:
        state = self._runs.stage(run_id, StageName.LESSON)
        return LessonPlan(**state.output) if state is not None and state.output else None

    def _singers(self, run_id: str, real: set[str]) -> dict[str, tuple[Asset, float]]:
        """For each shot a song's singer is laid over: her clip and where the shot starts in it."""
        state = self._runs.stage(run_id, StageName.LIPSYNC) if real else None
        if real and (state is None or state.status != StageStatus.DONE):
            raise ValueError(f"Run {run_id} has no finished lipsync stage ({state.status if state else 'not run'}); "
                             "run it first")
        output = state.output if state else {}
        found = {s: (self._assets.assets.get(a), output.get("offsets", {}).get(s, 0.0))
                 for s, a in output.get("clips", {}).items() if s in real}
        missing = sorted(s for s in real if found.get(s, (None,))[0] is None)
        if missing:
            raise ValueError(f"Run {run_id} cannot be assembled: no singer clip for {', '.join(missing)}")
        return found

    def _clips(self, run_id: str, board: Storyboard, real: set[str] = frozenset()) -> dict[str, tuple[Asset, StageName]]:
        """Each shot's clip and the stage it came from: lip-sync for sung shots, video for the rest. real: sung
        shots that are real footage from the video stage, with the singer laid over them."""
        stages = [StageName.VIDEO] + ([StageName.LIPSYNC] if any(s.kind == ShotKind.SUNG and s.id not in real
                                                                 for s in board.shots) else [])
        made = {}
        for name in stages:
            state = self._runs.stage(run_id, name)
            if state is None or state.status != StageStatus.DONE:
                raise ValueError(f"Run {run_id} has no finished {name} stage ({state.status if state else 'not run'});"
                                 f" run it first")
            made[name] = state.output.get("clips", {})
        clips, missing = {}, []
        for shot in board.shots:
            source = StageName.LIPSYNC if shot.kind == ShotKind.SUNG and shot.id not in real else StageName.VIDEO
            asset = self._assets.assets.get(made[source][shot.id]) if shot.id in made[source] else None
            if asset is None:
                missing.append(shot.id)
            else:
                clips[shot.id] = (asset, source)
        if missing:
            raise ValueError(f"Run {run_id} cannot be assembled: no clip for {', '.join(missing)}")
        return clips


def keyword_pops(plan: LessonPlan | None, words: list[TimedWord], shots: list[Shot]) -> list[Pop]:
    """The lesson's key words as the film shows them: each from the moment its line sings it (the line's start
    if the word cannot be timed), for KEYWORD_SECONDS or until the next key word, written as its key term when
    it is sung in pieces, in the top spot the subject of the shot on screen leaves clear."""
    if plan is None:
        return []
    spelling = {_NOT_LETTER.sub("", t.term.lower()): t.term for t in plan.key_terms}
    found = []
    for entry in plan.lines:
        sung = [w for w in words if w.line_index == entry.index]
        if not entry.keyword or not sung:
            continue
        at = sung_at(entry.keyword, sung)
        at = sung[0].start if at is None else at
        shot = next((s for s in shots if s.start <= at < s.end), None)
        spot = {Position.LEFT_THIRD: "right", Position.RIGHT_THIRD: "left"}.get(shot.position if shot else None, "centre")
        found.append((at, at + KEYWORD_SECONDS, spelling.get(_NOT_LETTER.sub("", entry.keyword.lower()), entry.keyword),
                      spot))
    found.sort()
    return [(at, min(until, later[0]) if later else until, text, spot)
            for (at, until, text, spot), later in zip(found, [*found[1:], None])]


def screen_words(shots: list[Shot], labels: bool = True) -> list[Screen]:
    """The words code draws over each shot, from the storyboard: a term's label and a number shot's figure,
    in the third the shot's main subject leaves open. labels: False when the film shows key words as they are
    sung, which take the labels' place."""
    found = []
    for shot in shots:
        texts = tuple(t for t in (shot.label if labels else None, shot.number_text) if t)
        if texts:
            side = "right" if shot.position == Position.LEFT_THIRD else "left"
            found.append((shot.start, shot.end, texts, side))
    return found


def frames_of(shots: list[Shot], i: int, fps: int) -> int:
    """Frames in shot i's slot, to the next shot's start (the last to its end), counted from the song's start
    so rounding never adds up into drift."""
    end = shots[i + 1].start if i + 1 < len(shots) else shots[i].end
    return round(end * fps) - round(shots[i].start * fps)
