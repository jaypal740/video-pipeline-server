import hashlib
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydantic import BaseModel

from src.core.configs import Settings
from src.repository.asset.asset_repository import AssetRepository
from src.repository.run.run_repository import RunRepository
from src.repository.song.song_repository import SongRepository
from src.schema.asset.asset_schema import Asset, ReviewState
from src.schema.base_schema import new_id, utcnow
from src.schema.enums import AssetKind, StageName, StageStatus
from src.schema.image.image_check_schema import Brief
from src.schema.run.stage_state_schema import StageState
from src.schema.storyboard.storyboard_schema import Place, Shot, ShotKind
from src.service.image.image_check_service import ImageCheckService
from src.service.lipsync.fal_lipsync_service import FalLipsyncService, LipsyncReply
from src.service.lipsync.lipsync_prices import lipsync_model
from src.service.llm.llm_service import LabelledImage, LlmCall
from src.service.local.ffmpeg_service import FfmpegService
from src.service.sheet.sheet_prompt import NO_OVERLAY_TEXT
from src.service.sheet.sheet_service import USABLE
from src.service.stage.stage_common import add_cost, asset_url, keep_best, storyboard_of
from src.service.storage.storage_service import Storage
from src.service.video.clip_checks import DURATION_SLACK, ClipChecks, ClipCheckService

logger = logging.getLogger(__name__)

JUDGE_USD = 0.002                                             # one frame plus the keyframe on Haiku
KEYFRAME_LABEL = "the keyframe it was drawn from"


class Attempt(BaseModel):
    """One lip-sync clip and its checks, made by a worker thread; the main thread turns it into Asset rows."""

    attempt: int
    seed: int
    reply: LipsyncReply
    data: bytes                                               # the clip with its audio stripped
    checks: ClipChecks
    code_problems: list[str]
    score: int | None
    asked_for: list[dict]
    notes: list[str]
    problems: list[str]
    check: LlmCall | None
    cache_key: str


def lipsync_brief(place: Place) -> Brief:
    """What the judge scores a singing frame against: the keyframe it was drawn from, kept as it was."""
    description = (f"Image 1 is a frame from a clip of the host singing his line; image 2 is the keyframe the clip was "
                   f"drawn from. The place is {place.look}, {place.lighting}. An open or moving mouth, a changed "
                   "expression and a small movement of the head or hands are expected while singing and never a fault.")
    fatal = [NO_OVERLAY_TEXT, f"No person or character that is not in the image labelled '{KEYFRAME_LABEL}'"]
    rules = ["The same person as the keyframe: the same face, hair and clothes",
             "The same place and framing as the keyframe", *fatal,
             "Nothing warped, melted together or broken; suitable for school children"]
    return Brief(description=description, rules=rules, fatal=fatal)


def code_problems(checks: ClipChecks, slot: float) -> list[str]:
    """The code checks that fit a singing face: long enough for the slot, no black frames. The motion rule
    is left out; a face that only sings moves too little for it."""
    found = []
    if checks.seconds < slot - DURATION_SLACK:
        found.append(f"the clip is {checks.seconds:.1f}s, the slot is {slot:.1f}s")
    if checks.black_frames:
        found.append(f"{checks.black_frames} black frames")
    return found


class LipsyncService:
    """Stage 5b: one clip per sung shot of the host singing his line, made by the lip-sync model from the
    shot's keyframe and the song from the shot's start. Checked by code (length, black frames) and then by
    the judge on the middle frame against the keyframe. Same rules as the clips: no person decides, the
    score decides, a seed is never drawn twice. The clip is stored silent; assembly lays the song on."""

    def __init__(self, run_repository: RunRepository, song_repository: SongRepository,
                 asset_repository: AssetRepository, storage: Storage, lipsync: FalLipsyncService,
                 ffmpeg: FfmpegService, checks: ClipCheckService, check: ImageCheckService, settings: Settings) -> None:
        self._runs = run_repository
        self._songs = song_repository
        self._assets = asset_repository
        self._storage = storage
        self._lipsync = lipsync
        self._ffmpeg = ffmpeg
        self._checks = checks
        self._check = check
        self._settings = settings

    def run(self, run_id: str, force: bool = False) -> dict[str, Asset]:
        board = storyboard_of(self._runs, run_id)
        keyframes = self._keyframes(run_id)
        model = self._settings.lipsync_model
        spec = lipsync_model(model)
        places = {p.id: p for p in board.places}
        sung = [s for s in board.shots if s.kind == ShotKind.SUNG]
        shots = [s for s in sung if s.repeat_of is None]
        plans, blocked = [], []
        for shot in shots:
            keyframe = keyframes.get(shot.id)
            if keyframe is None or keyframe.review not in USABLE:
                blocked.append(shot.id)
                continue
            slot = shot.end - shot.start
            plans.append((shot, keyframe, slot, spec.seconds(slot), lipsync_brief(places[shot.place])))
        key = hashlib.sha256("|".join([model, *(f"{s.id}:{k.sha256}:{s.start}:{n}" for s, k, _, n, _ in plans)])
                             .encode()).hexdigest()

        state = self._runs.stage(run_id, StageName.LIPSYNC)
        if state is not None and not force and state.status == StageStatus.DONE and state.input_hash == key:
            logger.info("lipsync %s: reused, inputs unchanged", run_id[:8])
            return {s: self._assets.assets.get(a) for s, a in state.output["clips"].items()}
        state = state or self._runs.stage_states.add(StageState(run_id=run_id, stage=StageName.LIPSYNC))
        state.status, state.attempts, state.error = StageStatus.RUNNING, state.attempts + 1, None
        state.input_hash, state.started_at, state.finished_at = key, utcnow(), None
        state = self._runs.stage_states.update(state)
        logger.info("lipsync %s: %s, %d sung clips to draw, %d blocked by keyframes, %d repeats reused", run_id[:8],
                    model, len(plans), len(blocked), len(sung) - len(shots))
        self._errors: dict[str, str] = {}
        try:
            made = self._draw_all(run_id, model, plans)
        except Exception as exc:
            state.status, state.error, state.finished_at = StageStatus.FAILED, str(exc), utcnow()
            self._runs.stage_states.update(state)
            logger.error("lipsync %s: failed: %s", run_id[:8], exc)
            raise
        for shot in sung:                                           # a chorus repeat shows its original's clip
            if shot.repeat_of in made:
                made[shot.id] = made[shot.repeat_of]
        self._finish(run_id, state, made, blocked, sung, len(plans))
        return made

    def _finish(self, run_id: str, state: StageState, made: dict[str, Asset], blocked: list[str], sung: list[Shot],
                drawn: int) -> None:
        used_best = sorted({a.id for a in made.values() if a.review == ReviewState.USED_BEST})
        blocked = sorted(set(blocked) | set(self._errors))
        missing = [s.id for s in sung if s.id not in made]
        state.status = StageStatus.FAILED if missing else StageStatus.DONE
        state.error = ("; ".join(f"{s}: {self._errors.get(s, 'no usable keyframe')}" for s in blocked)
                       or ("no clip for " + ", ".join(missing) if missing else None))
        state.output = {"clips": {s: a.id for s, a in made.items()}, "used_best": used_best, "blocked": blocked,
                        "errors": self._errors}
        state.finished_at = utcnow()
        self._runs.stage_states.update(state)
        retries = sum(a.params.get("try", 1) - 1 for s, a in made.items() if s == a.target_id)
        logger.log(logging.ERROR if missing else logging.INFO,
                   "lipsync %s: %s, %d clips, %d retries, %d used best, %d failed, $%.2f on the run so far, %.0fs",
                   run_id[:8], state.status, drawn, retries, len(used_best), len(blocked),
                   self._assets.run_cost(run_id), (state.finished_at - state.started_at).total_seconds())

    def _keep(self, shot_id: str, tries: list[Asset]) -> Asset | None:
        """Which of a shot's tries the stage uses (see keep_best). A clip the code checks failed has no score
        and is never used."""
        scored = [a for a in tries if a.params.get("score") is not None]
        return keep_best(self._assets, self._settings.image_floor_score, logger, "lipsync", shot_id, tries,
                         self._errors, scored, pool_only=True)

    # --- drawing ------------------------------------------------------------------------------------------

    def _draw_all(self, run_id: str, model: str, plans: list) -> dict[str, Asset]:
        made: dict[str, Asset] = {}
        todo = []
        for shot, keyframe, slot, audio_seconds, brief in plans:
            seeds = self._tries(model, shot, keyframe, audio_seconds)
            cached = {key: c for *_, key in seeds if (c := self._assets.find_cached(key)) is not None}
            tried = list(cached.values())
            usable = next((c for c in tried if c.review in USABLE), None)
            left = [s for s in seeds if s[2] not in cached]
            if usable is not None:
                logger.info("lipsync %s: reused %s (%s)", shot.id, usable.id[:8], usable.review)
                made[shot.id] = usable
            elif not left:
                self._keep(shot.id, tried)
            else:
                todo.append((shot, keyframe, slot, audio_seconds, brief, left))
        if not todo:
            return made
        self._guard_spend(run_id, model, [n for _, _, _, n, *_ in todo])
        song = self._songs.songs.get(self._runs.runs.get(run_id).song_id)
        images = {k.id: asset_url(self._storage, self._lipsync.upload, k) for _, k, *_ in todo}
        frames = {k.id: self._storage.get(k.path) for _, k, *_ in todo}
        audio = {s.id: self._lipsync.upload(self._cut(song.audio_path, s.start, n), "audio/mp4")
                 for s, _, _, n, *_ in todo}
        with ThreadPoolExecutor(max_workers=self._settings.video_workers) as pool:
            futures = {pool.submit(self._attempts, model, images[job[1].id], audio[job[0].id], frames[job[1].id],
                                   *job): job for job in todo}
            for future in as_completed(futures):
                shot, keyframe, _, audio_seconds, *_ = futures[future]
                try:
                    tries = future.result()
                except Exception as exc:
                    logger.error("lipsync %s: error: %s", shot.id, exc)
                    self._errors[shot.id] = str(exc)
                    continue
                kept = self._store_attempts(run_id, shot, keyframe, audio_seconds, tries)
                if kept is not None:
                    made[shot.id] = kept
        return made

    def _guard_spend(self, run_id: str, model: str, seconds: list[float]) -> None:
        spec = lipsync_model(model)
        first = sum(spec.price_per_s * n + JUDGE_USD for n in seconds)
        worst = first * self._settings.image_max_tries
        spent = self._assets.run_cost(run_id)
        logger.info("lipsync %s: %d clips, %.1f seconds of audio at $%.3f/s, about $%.2f first time, at most $%.2f "
                    "with retries; $%.2f spent on the run so far, cap $%.2f", run_id[:8], len(seconds), sum(seconds),
                    spec.price_per_s, first, worst, spent, self._settings.spend_cap_usd)
        if spent + worst > self._settings.spend_cap_usd:
            raise ValueError(f"lip-syncing {len(seconds)} clips could take run {run_id[:8]} to ${spent + worst:.2f}, "
                             f"past the cap of ${self._settings.spend_cap_usd:.2f}; raise PIPELINE_SPEND_CAP_USD "
                             "or reduce the run")

    def _attempts(self, model: str, image_url: str, audio_url: str, keyframe_png: bytes, shot: Shot, keyframe: Asset,
                  slot: float, audio_seconds: float, brief: Brief, seeds: list[tuple[int, int, str]]) -> list[Attempt]:
        """Network only: draw, strip the audio, check by code, judge the middle frame, retry on a new seed."""
        tries = []
        for attempt, seed, cache_key in seeds:
            reply = self._lipsync.generate(model, image_url, audio_url, audio_seconds, seed=seed)
            data = self._silent(reply.data)
            checks = self._checks.check(data, wanted_seconds=slot)
            problems = code_problems(checks, slot)
            found = list(problems)
            score, asked_for, notes, call = None, [], list(problems), None
            if not problems:
                middle = self._checks.frames(data, min(checks.seconds, slot))[1]   # the middle of what is shown
                refs = [LabelledImage(label=KEYFRAME_LABEL, data=keyframe_png)]
                checklist, call = self._check.check(middle, refs, brief)
                score, asked_for = checklist.score, [a.model_dump() for a in checklist.asked_for]
                problems, notes = self._check.problems(checklist, brief), self._check.notes(checklist)
            tries.append(Attempt(attempt=attempt, seed=seed, reply=reply, data=data, checks=checks, code_problems=found,
                                 score=score, asked_for=asked_for, notes=notes, problems=problems, check=call,
                                 cache_key=cache_key))
            if not problems:
                logger.info("lipsync %s: passed on try %d, $%.3f, %.0fs", shot.id, attempt, reply.cost_usd, reply.seconds)
                return tries
            logger.warning("lipsync %s: try %d failed %d checks:\n%s", shot.id, attempt, len(problems),
                           "\n".join(f"  - {p}" for p in problems))
        logger.warning("lipsync %s: no try passed after %d", shot.id, len(seeds))
        return tries

    def _store_attempts(self, run_id: str, shot: Shot, keyframe: Asset, audio_seconds: float,
                        tries: list[Attempt]) -> Asset | None:
        stored = []
        for t in tries:
            asset_id = new_id()
            path = f"runs/{run_id}/lipsync/{shot.id}/{t.attempt}-{asset_id[:8]}.mp4"
            self._storage.put(path, t.data, "video/mp4")
            asset = self._assets.assets.add(Asset(
                id=asset_id, run_id=run_id, stage=StageName.LIPSYNC, kind=AssetKind.VIDEO, path=path,
                sha256=hashlib.sha256(t.data).hexdigest(), target_id=shot.id, cache_key=t.cache_key,
                provider=self._lipsync.name, model=t.reply.model, seed=t.reply.seed,
                params={"try": t.attempt, "keyframe": keyframe.id, "seconds": t.checks.seconds,
                        "audio_start": shot.start, "audio_seconds": audio_seconds, "fps": t.checks.fps,
                        "width": t.checks.width, "height": t.checks.height, "request_id": t.reply.request_id,
                        "wall": t.reply.seconds, "checks": t.code_problems, "score": t.score, "asked_for": t.asked_for},
                cost_usd=t.reply.cost_usd, review=ReviewState.PASSED if not t.problems else ReviewState.FAILED,
                review_notes="\n".join(t.notes) or None))
            add_cost(self._assets, run_id, StageName.LIPSYNC, self._lipsync.name, t.reply.model, t.reply.cost_usd,
                     shot.id, audio_seconds, "second")
            if t.check is not None:
                add_cost(self._assets, run_id, StageName.LIPSYNC, t.check.provider, t.check.model, t.check.cost_usd,
                         shot.id, t.check.input_tokens + t.check.output_tokens, "tokens")
            stored.append(asset)
        return self._keep(shot.id, stored)

    # --- small helpers ------------------------------------------------------------------------------------

    def _cut(self, audio_path: str, start: float, seconds: float) -> bytes:
        """The song from `start`, `seconds` long, as AAC in mp4; padded with silence past the song's end so the
        model always gets its minimum."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "cut.m4a"
            self._ffmpeg.run(["-ss", f"{start:.3f}", "-i", audio_path, "-vn", "-af", "apad", "-t", f"{seconds:.3f}",
                              "-c:a", "aac", "-b:a", "192k", "-f", "ipod", str(out)])
            return out.read_bytes()

    def _silent(self, data: bytes) -> bytes:
        """The clip without its audio track; the picture is copied, not re-encoded."""
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "in.mp4", Path(tmp) / "out.mp4"
            src.write_bytes(data)
            self._ffmpeg.run(["-i", str(src), "-an", "-c:v", "copy", str(out)])
            return out.read_bytes()

    def _tries(self, model: str, shot: Shot, keyframe: Asset, audio_seconds: float) -> list[tuple[int, int, str]]:
        """A shot's tries in order: try number, seed and cache key, from the keyframe and the audio sent."""
        what = f"{keyframe.sha256}|{shot.start}|{audio_seconds}"
        base = int(hashlib.sha256(what.encode()).hexdigest()[:8], 16) % 2**31
        return [(i + 1, base + i, hashlib.sha256(f"{model}|{what}|{base + i}".encode()).hexdigest())
                for i in range(self._settings.image_max_tries)]

    def _keyframes(self, run_id: str) -> dict[str, Asset]:
        state = self._runs.stage(run_id, StageName.KEYFRAMES)
        if state is None or state.status != StageStatus.DONE:
            raise ValueError(f"Run {run_id} has no finished keyframes (keyframes stage: "
                             f"{state.status if state else 'not run'}); run the keyframes stage first")
        return {s: a for s, i in state.output["keyframes"].items() if (a := self._assets.assets.get(i)) is not None}
