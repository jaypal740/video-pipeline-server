import hashlib
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydantic import BaseModel

from src.core.configs import Settings
from src.repository.asset.asset_repository import AssetRepository
from src.repository.run.run_repository import RunRepository
from src.schema.asset.asset_schema import Asset, ReviewState
from src.schema.base_schema import new_id, utcnow
from src.schema.enums import AssetKind, StageName, StageStatus
from src.schema.image.image_check_schema import Brief
from src.schema.run.stage_state_schema import StageState
from src.schema.storyboard.storyboard_schema import Moves, Shot, ShotKind
from src.service.image.image_check_service import ImageCheckService
from src.service.lesson.lesson_rules import band_rules
from src.service.llm.llm_service import LabelledImage, LlmCall, LlmService
from src.service.local.ffmpeg_service import FfmpegService
from src.service.sheet.sheet_prompt import Style
from src.service.sheet.sheet_service import USABLE
from src.service.stage.stage_common import (PASSED, StopLoss, add_cost, asset_url, keep_best, storyboard_of,
                                            write_correction)
from src.service.storage.storage_service import Storage
from src.service.video.camera_move import move_filter
from src.service.video.clip_checks import ClipCheckService, ClipChecks
from src.service.video.clip_prompt import NEGATIVE, clip_brief, clip_prompt, host_look_for
from src.service.video.fal_video_service import FalVideoService, VideoReply
from src.service.video.video_prices import video_model

logger = logging.getLogger(__name__)

JUDGE_USD = 0.003                                             # three frames plus the keyframe on Haiku


class Attempt(BaseModel):
    """One clip and its checks, made by a worker thread; the main thread turns it into Asset rows."""

    attempt: int
    seed: int
    prompt: str
    correction: str | None = None
    fix: LlmCall | None = None
    reply: VideoReply
    checks: ClipChecks
    score: int | None
    asked_for: list[dict]
    notes: list[str]
    problems: list[str]
    check: LlmCall | None
    cache_key: str


class VideoService:
    """Stage 5a: one clip per shot. A shot where something moves is animated by the video model between its
    start and end keyframes, at the slot's own length, then checked by code (does it move, does it freeze)
    and by the judge on three frames; its audio is dropped, since the song is the only sound. A shot where
    only the camera moves is an ffmpeg move over its keyframe. A sung shot is left to the lip-sync stage.
    Same rules as the keyframes: no person decides, the score decides, a retry carries a correction, a seed
    is never drawn twice."""

    def __init__(self, run_repository: RunRepository, asset_repository: AssetRepository, storage: Storage,
                 video: FalVideoService, checks: ClipCheckService, check: ImageCheckService, llm: LlmService,
                 ffmpeg: FfmpegService, settings: Settings) -> None:
        self._runs = run_repository
        self._assets = asset_repository
        self._storage = storage
        self._video = video
        self._checks = checks
        self._ffmpeg = ffmpeg
        self._check = check
        self._llm = llm
        self._settings = settings

    def run(self, run_id: str, force: bool = False) -> dict[str, Asset]:
        board = storyboard_of(self._runs, run_id)
        keyframes, ends = self._keyframes(run_id), self._end_keyframes(run_id)
        style, model = Style.of(band_rules(board.grade_band)), self._settings.video_model
        spec = video_model(model)
        cast = {c.id: c for c in board.cast}
        places = {p.id: p for p in board.places}
        shots = [s for s in board.shots if s.repeat_of is None]
        plans, moves, blocked = [], [], []
        for shot in shots:
            keyframe = keyframes.get(shot.id)
            if keyframe is None or keyframe.review not in USABLE:
                blocked.append(shot.id)
                continue
            if shot.kind == ShotKind.SUNG and self._settings.video_sung_mode == "skip":
                continue                                        # the lip-sync stage makes this one
            if shot.kind != ShotKind.SUNG and shot.moves == Moves.CAMERA:
                moves.append((shot, keyframe))
                continue
            subjects = [cast[s] for s in shot.subjects]
            prompt = clip_prompt(style, shot, places[shot.place], subjects, host_look_for(subjects))
            seconds = spec.seconds((shot.end - shot.start) * self._settings.video_length_factor)
            end = ends.get(shot.id)
            plans.append((shot, prompt, keyframe, seconds, clip_brief(shot, places[shot.place], subjects), end))
        key = hashlib.sha256("|".join([model, *(f"{s.id}:{p}:{k.sha256}:{e.sha256 if e else ''}:{n}"
                                                for s, p, k, n, _, e in plans),
                                       *(f"{s.id}:camera:{k.sha256}:{self._move_key(s)}" for s, k in moves)])
                             .encode()).hexdigest()

        state = self._runs.stage(run_id, StageName.VIDEO)
        if state is not None and not force and state.status == StageStatus.DONE and state.input_hash == key:
            logger.info("video %s: reused, inputs unchanged", run_id[:8])
            return {s: self._assets.assets.get(a) for s, a in state.output["clips"].items()}
        state = state or self._runs.stage_states.add(StageState(run_id=run_id, stage=StageName.VIDEO))
        state.status, state.attempts, state.error = StageStatus.RUNNING, state.attempts + 1, None
        state.input_hash, state.started_at, state.finished_at = key, utcnow(), None
        state = self._runs.stage_states.update(state)
        logger.info("video %s: %s, %d clips to draw (%d with an end frame), %d camera moves, %d blocked by "
                    "keyframes, %d repeats reused", run_id[:8], model, len(plans), sum(1 for p in plans if p[-1]),
                    len(moves), len(blocked), len(board.shots) - len(shots))
        self._errors: dict[str, str] = {}
        self._stop = StopLoss(self._settings.image_stop_loss, "clips")
        try:
            made = {s.id: self._camera(run_id, s, k) for s, k in moves}
            made.update(self._draw_all(run_id, model, plans))
        except Exception as exc:
            state.status, state.error, state.finished_at = StageStatus.FAILED, str(exc), utcnow()
            self._runs.stage_states.update(state)
            logger.error("video %s: failed: %s", run_id[:8], exc)
            raise
        for shot in board.shots:                                    # a chorus repeat shows its original's clip
            if shot.repeat_of in made:
                made[shot.id] = made[shot.repeat_of]
        self._finish(run_id, state, made, blocked, len(plans))
        return made

    def _finish(self, run_id: str, state: StageState, made: dict[str, Asset], blocked: list[str], drawn: int) -> None:
        used_best = sorted({a.id for a in made.values() if a.review == ReviewState.USED_BEST})
        blocked = sorted(set(blocked) | set(self._errors))
        state.status = StageStatus.FAILED if blocked else StageStatus.DONE
        state.error = ("; ".join(f"{s}: {self._errors.get(s, 'no usable keyframe')}" for s in blocked) or None)
        state.output = {"clips": {s: a.id for s, a in made.items()}, "used_best": used_best, "blocked": blocked,
                        "errors": self._errors}
        state.finished_at = utcnow()
        self._runs.stage_states.update(state)
        retries = sum(a.params.get("try", 1) - 1 for s, a in made.items() if s == a.target_id)
        logger.log(logging.ERROR if blocked else logging.INFO,
                   "video %s: %s, %d clips, %d retries, %d used best, %d failed, $%.2f on the run so far, %.0fs",
                   run_id[:8], state.status, drawn, retries, len(used_best), len(blocked),
                   self._assets.run_cost(run_id), (state.finished_at - state.started_at).total_seconds())

    def recheck(self, run_id: str) -> dict[str, Asset]:
        """Run the code checks and the judge again on every clip of this run that did not pass, with the
        current rules; nothing is drawn. The stage then keeps the best try of each shot or fails as before."""
        board = storyboard_of(self._runs, run_id)
        state = self._runs.stage(run_id, StageName.VIDEO)
        if state is None or not state.output:
            raise ValueError(f"Run {run_id} has no clips yet")
        keyframes = self._keyframes(run_id)
        cast = {c.id: c for c in board.cast}
        places = {p.id: p for p in board.places}
        shots = {s.id: s for s in board.shots if s.repeat_of is None}
        tries = [t for shot_id in shots if (owner := self._owner(state, shot_id)) is not None
                 for t in self._assets.assets.list(run_id=owner, stage=StageName.VIDEO, target_id=shot_id)]
        self._errors = {k: v for k, v in state.output.get("errors", {}).items() if k not in shots}
        for asset in tries:
            if asset.review in PASSED:
                continue
            shot = shots[asset.target_id]
            subjects = [cast[s] for s in shot.subjects]
            brief = clip_brief(shot, places[shot.place], subjects)
            data = self._storage.get(asset.path)
            checks = self._checks.check(data, asset.params["seconds_requested"])
            score, asked_for, notes, problems = None, [], list(checks.problems), list(checks.problems)
            if not checks.problems:
                first, middle, last = self._checks.frames(data, checks.seconds)
                keyframe = keyframes.get(shot.id)
                refs = [LabelledImage(label="frame 1 of 3, the first", data=first),
                        LabelledImage(label="frame 3 of 3, the last", data=last),
                        LabelledImage(label="the keyframe it was animated from", data=self._storage.get(keyframe.path))]
                checklist, call = self._check.check(middle, refs, brief)
                add_cost(self._assets, run_id, StageName.VIDEO, call.provider, call.model, call.cost_usd, shot.id,
                         call.input_tokens + call.output_tokens, "tokens")
                score, asked_for = checklist.score, [a.model_dump() for a in checklist.asked_for]
                problems, notes = self._check.problems(checklist, brief), self._check.notes(checklist)
            asset.params = {**asset.params, "score": score, "asked_for": asked_for, "checks": checks.problems,
                            "motion": checks.motion, "tail_motion": checks.tail_motion,
                            "active_seconds": checks.active_seconds}
            asset.review = ReviewState.PASSED if not problems else ReviewState.FAILED
            asset.review_notes = "\n".join(notes) or None
            self._assets.assets.update(asset)
            logger.info("clip %s try %d rechecked: %s, %s/10", shot.id, asset.params["try"], asset.review, score)
        made: dict[str, Asset] = {}
        by_shot: dict[str, list[Asset]] = {}
        for asset in tries:
            by_shot.setdefault(asset.target_id, []).append(asset)
        for shot_id, shot_tries in by_shot.items():
            kept = self._keep(shot_id, shot_tries)
            if kept is not None:
                made[shot_id] = kept
        for shot in board.shots:
            if shot.repeat_of in made:
                made[shot.id] = made[shot.repeat_of]
        blocked = [s for s in state.output.get("blocked", []) if s not in by_shot]
        state.started_at = state.started_at or utcnow()
        self._finish(run_id, state, made, blocked, len(by_shot))
        return made

    def _owner(self, state: StageState, shot_id: str) -> str | None:
        """The run whose rows hold this shot's tries: the run that drew the kept clip, else this run."""
        kept = self._assets.assets.get(state.output["clips"].get(shot_id, ""))
        if kept is not None:
            return kept.run_id
        return state.run_id if shot_id in state.output.get("blocked", []) else None

    def _keep(self, shot_id: str, tries: list[Asset]) -> Asset | None:
        """Which of a shot's tries the stage uses (see keep_best). A clip the code checks failed has no score
        and is never used."""
        scored = [a for a in tries if a.params.get("score") is not None]
        return keep_best(self._assets, self._settings.image_floor_score, logger, "clip", shot_id, tries,
                         self._errors, scored, pool_only=True)

    # --- drawing ------------------------------------------------------------------------------------------

    def _draw_all(self, run_id: str, model: str, plans: list) -> dict[str, Asset]:
        made: dict[str, Asset] = {}
        todo = []
        for shot, prompt, keyframe, seconds, brief, end in plans:
            seeds = self._tries(model, prompt, keyframe, seconds, end)
            cached = {key: c for *_, key in seeds if (c := self._assets.find_cached(key)) is not None}
            tried = list(cached.values())
            usable = next((c for c in tried if c.review in USABLE), None)
            left = [s for s in seeds if s[2] not in cached]
            if usable is not None:
                logger.info("clip %s: reused %s (%s)", shot.id, usable.id[:8], usable.review)
                made[shot.id] = usable
            elif not left:
                self._keep(shot.id, tried)
            else:
                last = max(tried, key=lambda c: c.params.get("try", 1), default=None)
                findings = (last.review_notes or "").splitlines() if last is not None else []
                todo.append((shot, prompt, keyframe, seconds, brief, left, findings, end))
        if not todo:
            return made
        self._guard_spend(run_id, model, [n for _, _, _, n, *_ in todo])
        urls = {k.id: asset_url(self._storage, self._video.upload, k) for _, _, k, *_ in todo}
        urls.update({e.id: asset_url(self._storage, self._video.upload, e) for *_, e in todo if e is not None})
        frames = {k.id: self._storage.get(k.path) for _, _, k, *_ in todo}
        with ThreadPoolExecutor(max_workers=self._settings.video_workers) as pool:
            futures = {pool.submit(self._attempts, model, urls[k.id], frames[k.id], *job[:-1],
                                   urls[e.id] if e is not None else None): job
                       for job in todo for k, e in [(job[2], job[-1])]}
            for future in as_completed(futures):
                shot, _, keyframe, *_, end = futures[future]
                try:
                    tries = future.result()
                except Exception as exc:
                    logger.error("clip %s: error: %s", shot.id, exc)
                    self._errors[shot.id] = str(exc)
                    continue
                if not tries:
                    self._errors[shot.id] = self._stop.reason
                    continue
                kept = self._store_attempts(run_id, shot, keyframe, tries, end)
                if kept is not None:
                    made[shot.id] = kept
        if self._stop.reason:
            logger.error("video %s: %s; %d clips not drawn", run_id[:8], self._stop.reason,
                         sum(1 for e in self._errors.values() if e == self._stop.reason))
        return made

    def _guard_spend(self, run_id: str, model: str, seconds: list[int]) -> None:
        spec = video_model(model)
        first = sum(spec.price_per_s * n + JUDGE_USD for n in seconds)
        worst = first * self._settings.image_max_tries
        spent = self._assets.run_cost(run_id)
        logger.info("video %s: %d clips, %d seconds at $%.3f/s, about $%.2f first time, at most $%.2f with retries; "
                    "$%.2f spent on the run so far, cap $%.2f", run_id[:8], len(seconds), sum(seconds), spec.price_per_s,
                    first, worst, spent, self._settings.spend_cap_usd)
        if spent + worst > self._settings.spend_cap_usd:
            raise ValueError(f"drawing {len(seconds)} clips could take run {run_id[:8]} to ${spent + worst:.2f}, "
                             f"past the cap of ${self._settings.spend_cap_usd:.2f}; raise PIPELINE_SPEND_CAP_USD "
                             "or reduce the run")

    def _attempts(self, model: str, url: str, keyframe_png: bytes, shot: Shot, prompt: str, keyframe: Asset,
                  seconds: float, brief: Brief, seeds: list[tuple[int, int, str]], findings: list[str],
                  end_url: str | None) -> list[Attempt]:
        """Network only: draw between the start and end frames, drop the audio, check by code, judge on three
        frames, retry on a new seed with a correction."""
        if self._stop.reason:
            return []
        tries = []
        for attempt, seed, cache_key in seeds:
            correction, fix = (write_correction(self._llm, self._settings.image_check_model, logger, "clip", shot.id,
                                                prompt, brief, findings) if findings else (None, None))
            drawn = f"{prompt} {correction}" if correction else prompt
            reply = self._video.generate(model, drawn, url, seconds, seed=seed, negative=NEGATIVE, end_image_url=end_url)
            reply.data = self._silent(reply.data)               # the models make up speech; the song is the sound
            checks = self._checks.check(reply.data, reply.seconds_requested)
            score, asked_for, notes, call = None, [], list(checks.problems), None
            if not checks.problems:
                first, middle, last = self._checks.frames(reply.data, checks.seconds)
                refs = [LabelledImage(label="frame 1 of 3, the first", data=first),
                        LabelledImage(label="frame 3 of 3, the last", data=last),
                        LabelledImage(label="the keyframe it was animated from", data=keyframe_png)]
                checklist, call = self._check.check(middle, refs, brief)
                score, asked_for = checklist.score, [a.model_dump() for a in checklist.asked_for]
                problems, notes = self._check.problems(checklist, brief), self._check.notes(checklist)
            else:
                problems = list(checks.problems)
            findings = notes
            tries.append(Attempt(attempt=attempt, seed=seed, prompt=drawn, correction=correction, fix=fix, reply=reply,
                                 checks=checks, score=score, asked_for=asked_for, notes=notes, problems=problems,
                                 check=call, cache_key=cache_key))
            if not problems:
                logger.info("clip %s: passed on try %d, $%.3f, %.0fs", shot.id, attempt, reply.cost_usd, reply.seconds)
                self._stop.count(passed=True)
                return tries
            logger.warning("clip %s: try %d failed %d checks:\n%s", shot.id, attempt, len(problems),
                           "\n".join(f"  - {p}" for p in problems))
        logger.warning("clip %s: no try passed after %d", shot.id, len(seeds))
        self._stop.count(passed=False)
        return tries

    def _store_attempts(self, run_id: str, shot: Shot, keyframe: Asset, tries: list[Attempt],
                        end: Asset | None) -> Asset | None:
        stored = []
        for t in tries:
            asset_id = new_id()
            path = f"runs/{run_id}/clips/{shot.id}/{t.attempt}-{asset_id[:8]}.mp4"
            self._storage.put(path, t.reply.data, t.reply.content_type)
            asset = self._assets.assets.add(Asset(
                id=asset_id, run_id=run_id, stage=StageName.VIDEO, kind=AssetKind.VIDEO, path=path,
                sha256=hashlib.sha256(t.reply.data).hexdigest(), target_id=shot.id, cache_key=t.cache_key,
                provider=self._video.name, model=t.reply.model, seed=t.reply.seed, prompt=t.prompt,
                params={"try": t.attempt, "route": "model", "keyframe": keyframe.id,
                        "end_keyframe": end.id if end is not None else None,
                        "seconds_requested": t.reply.seconds_requested,
                        "seconds": t.checks.seconds, "fps": t.checks.fps, "width": t.checks.width,
                        "height": t.checks.height, "request_id": t.reply.request_id, "wall": t.reply.seconds,
                        "motion": t.checks.motion, "tail_motion": t.checks.tail_motion, "checks": t.checks.problems,
                        "active_seconds": t.checks.active_seconds,
                        "score": t.score, "asked_for": t.asked_for, "correction": t.correction},
                cost_usd=t.reply.cost_usd, review=ReviewState.PASSED if not t.problems else ReviewState.FAILED,
                review_notes="\n".join(t.notes) or None))
            add_cost(self._assets, run_id, StageName.VIDEO, self._video.name, t.reply.model, t.reply.cost_usd, shot.id,
                     t.reply.seconds_requested, "second")
            for call in (c for c in (t.check, t.fix) if c is not None):
                add_cost(self._assets, run_id, StageName.VIDEO, call.provider, call.model, call.cost_usd, shot.id,
                         call.input_tokens + call.output_tokens, "tokens")
            stored.append(asset)
        return self._keep(shot.id, stored)

    # --- small helpers ------------------------------------------------------------------------------------

    def _tries(self, model: str, prompt: str, keyframe: Asset, seconds: float,
               end: Asset | None) -> list[tuple[int, int, str]]:
        """A shot's tries in order: try number, seed and cache key. The key uses the shot's own prompt, not a
        retry's correction, so a corrected retry is still found and never drawn twice."""
        frames = f"{keyframe.sha256}|{end.sha256}" if end is not None else keyframe.sha256
        base = int(hashlib.sha256(f"{prompt}|{frames}".encode()).hexdigest()[:8], 16) % 2**31
        return [(i + 1, base + i, hashlib.sha256(f"{model}|{prompt}|{frames}|{base + i}|{seconds}".encode())
                 .hexdigest()) for i in range(self._settings.image_max_tries)]

    # --- camera moves and audio ---------------------------------------------------------------------------

    def _move_key(self, shot: Shot) -> str:
        s = self._settings
        return (f"{shot.camera.move}:{shot.camera.direction}:{shot.camera.speed}:{self._move_frames(shot)}:"
                f"{s.assembly_width}x{s.assembly_height}@{s.assembly_fps}")

    def _move_frames(self, shot: Shot) -> int:
        return max(1, round((shot.end - shot.start) * self._settings.assembly_fps))

    def _camera(self, run_id: str, shot: Shot, keyframe: Asset) -> Asset:
        """The clip of a shot where only the camera moves: ffmpeg over the keyframe. Nothing to judge, since
        the picture is the approved keyframe; the same move on the same keyframe is made once."""
        cache_key = hashlib.sha256(f"camera|{keyframe.sha256}|{self._move_key(shot)}".encode()).hexdigest()
        cached = self._assets.find_cached(cache_key)
        if cached is not None:
            logger.info("clip %s: reused camera move %s", shot.id, cached.id[:8])
            return cached
        s, frames = self._settings, self._move_frames(shot)
        with tempfile.TemporaryDirectory() as tmp:
            still, clip = Path(tmp) / "keyframe.png", Path(tmp) / "clip.mp4"
            still.write_bytes(self._storage.get(keyframe.path))
            self._ffmpeg.run(["-loop", "1", "-i", str(still), "-vf",
                              move_filter(shot, frames, s.assembly_width, s.assembly_height, s.assembly_fps),
                              "-frames:v", str(frames), "-c:v", "libx264", "-crf", "18", str(clip)])
            data = clip.read_bytes()
        asset_id = new_id()
        path = f"runs/{run_id}/clips/{shot.id}/camera-{asset_id[:8]}.mp4"
        self._storage.put(path, data, "video/mp4")
        logger.info("clip %s: camera %s made by ffmpeg, %d frames", shot.id, shot.camera.move, frames)
        return self._assets.assets.add(Asset(
            id=asset_id, run_id=run_id, stage=StageName.VIDEO, kind=AssetKind.VIDEO, path=path,
            sha256=hashlib.sha256(data).hexdigest(), target_id=shot.id, cache_key=cache_key, provider="ffmpeg",
            model="ffmpeg", prompt=None,
            params={"try": 1, "route": "camera", "keyframe": keyframe.id, "move": shot.camera.move,
                    "seconds": frames / s.assembly_fps, "frames": frames},
            cost_usd=0.0, review=ReviewState.PASSED))

    def _silent(self, data: bytes) -> bytes:
        """The clip with its audio track removed; the picture is copied untouched."""
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "in.mp4", Path(tmp) / "out.mp4"
            src.write_bytes(data)
            self._ffmpeg.run(["-i", str(src), "-an", "-c:v", "copy", str(out)])
            return out.read_bytes()

    def _keyframes(self, run_id: str) -> dict[str, Asset]:
        state = self._runs.stage(run_id, StageName.KEYFRAMES)
        if state is None or state.status != StageStatus.DONE:
            raise ValueError(f"Run {run_id} has no finished keyframes (keyframes stage: "
                             f"{state.status if state else 'not run'}); run the keyframes stage first")
        return {s: a for s, i in state.output["keyframes"].items() if (a := self._assets.assets.get(i)) is not None}

    def _end_keyframes(self, run_id: str) -> dict[str, Asset]:
        """Each shot's planned last frame, from the keyframes stage; a shot without one is drawn from its start."""
        state = self._runs.stage(run_id, StageName.KEYFRAMES)
        return {s: a for s, i in (state.output or {}).get("end_keyframes", {}).items()
                if (a := self._assets.assets.get(i)) is not None}
