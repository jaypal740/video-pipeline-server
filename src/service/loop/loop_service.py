import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from src.core.configs import Settings
from src.repository.asset.asset_repository import AssetRepository
from src.repository.run.run_repository import RunRepository
from src.schema.asset.asset_schema import Asset, ReviewState
from src.schema.base_schema import new_id, utcnow
from src.schema.enums import AssetKind, StageName, StageStatus
from src.schema.image.image_check_schema import Brief
from src.schema.lesson.lesson_schema import LessonPlan
from src.schema.run.stage_state_schema import StageState
from src.schema.storyboard.script_schema import AnimationScript, Drawing
from src.service.drawing.drawing_prompt import must_not_show
from src.service.image.image_check_service import ImageCheckService
from src.service.llm.llm_service import LabelledImage, LlmCall, LlmService
from src.service.loop.loop_checks import LoopCheck, LoopCutService
from src.service.loop.loop_prompt import LOOP_SECONDS, loop_brief, loop_prompt
from src.service.stage.stage_common import add_cost, add_llm_cost, asset_url, fail_stage, script_of, write_correction
from src.service.storage.storage_service import Storage
from src.service.video.fal_video_service import FalVideoService, VideoReply

logger = logging.getLogger(__name__)

STAGE = StageName.LOOPS
USABLE = (ReviewState.PASSED, ReviewState.APPROVED, ReviewState.USED_BEST)


class LoopService:
    """Stage 5: every drawing that has a motion, animated once into a short loop the render repeats. The
    video model draws from the drawing back to the same drawing, still on the key colour, so each frame is
    cut out the way the drawing was. Code checks that every frame cuts out, that the loop ends as it began
    and that something moves; the judge checks the frame furthest from the drawing against the drawing. A
    loop that fails is made again on a new seed, told what the checks found wrong; when that fails too the
    better try is used (used_best). Only a drawing no loop of which could be cut out fails the stage. A
    re-run with the same drawings reuses the stored loops."""

    def __init__(self, run_repository: RunRepository, asset_repository: AssetRepository, storage: Storage,
                 video: FalVideoService, cutter: LoopCutService, check: ImageCheckService, llm: LlmService,
                 settings: Settings) -> None:
        self._runs = run_repository
        self._assets = asset_repository
        self._storage = storage
        self._video = video
        self._cutter = cutter
        self._check = check
        self._llm = llm
        self._settings = settings

    def run(self, run_id: str, until: float | None = None, force: bool = False) -> dict[str, Asset]:
        """until: animate only the drawings the beats starting before that second of the song show."""
        if self._runs.runs.get(run_id) is None:
            raise LookupError(f"Run {run_id} not found")
        board = script_of(self._runs, run_id)
        avoid = {b.id: b.avoid for b in LessonPlan(**self._runs.stage(run_id, StageName.LESSON).output).beats}
        shown = {i for b in board.beats if until is None or b.start < until for a in b.actions for i in (a.drawing, a.to)}
        todo = [d for d in board.drawings if d.motion and d.id in shown]
        stills = self._stills(run_id, todo)
        model = self._settings.video_model
        key = hashlib.sha256("|".join([model] + [f"{d.id}:{stills[d.id].sha256}:{loop_prompt(d)}" for d in todo])
                             .encode()).hexdigest()

        state = self._runs.stage(run_id, STAGE)
        if state is not None and not force and state.status == StageStatus.DONE and state.input_hash == key:
            logger.info("loops %s: reused, inputs unchanged", run_id[:8])
            return {name: self._assets.assets.get(a) for name, a in state.output["loops"].items()}
        state = state or self._runs.stage_states.add(StageState(run_id=run_id, stage=STAGE))
        state.status, state.attempts, state.error = StageStatus.RUNNING, state.attempts + 1, None
        state.input_hash, state.started_at, state.finished_at = key, utcnow(), None
        state = self._runs.stage_states.update(state)
        logger.info("loops %s: %s, %d of %d drawings to animate", run_id[:8], model, len(todo), len(board.drawings))
        try:
            made = self._animate_all(run_id, model, board, todo, stills, avoid)
        except Exception as exc:
            fail_stage(self._runs, state, exc, logger, "loops")
            raise
        failed = [name for name, a in made.items() if a.review == ReviewState.FAILED]
        state.status = StageStatus.FAILED if failed else StageStatus.DONE
        state.error = "; ".join(f"{name}: no loop could be cut out: {made[name].review_notes}" for name in failed) or None
        state.output = {"loops": {name: a.id for name, a in made.items()}, "until": until,
                        "used_best": [name for name, a in made.items() if a.review == ReviewState.USED_BEST]}
        state.finished_at = utcnow()
        self._runs.stage_states.update(state)
        logger.log(logging.ERROR if failed else logging.INFO,
                   "loops %s: %s, %d loops, %d retries, %d used best, %d failed, $%.2f on the run so far, %.0fs",
                   run_id[:8], state.status, len(made), sum(a.params.get("try", 1) - 1 for a in made.values()),
                   len(state.output["used_best"]), len(failed), self._assets.run_cost(run_id),
                   (state.finished_at - state.started_at).total_seconds())
        return made

    def _stills(self, run_id: str, todo: list[Drawing]) -> dict[str, Asset]:
        """The drawing each loop is made from, as the drawings stage left it."""
        drawn = self._runs.stage(run_id, StageName.VISUAL_IDENTITY)
        stills = {name: self._assets.assets.get(a) for name, a in (drawn.output.get("drawings", {}) if drawn else {}).items()}
        missing = [d.id for d in todo if d.id not in stills or stills[d.id].review not in USABLE]
        if missing:
            raise ValueError(f"Run {run_id} has no usable drawing for {', '.join(missing)}; run the drawings stage first")
        return stills

    def _animate_all(self, run_id: str, model: str, board: AnimationScript, todo: list[Drawing],
                     stills: dict[str, Asset], avoid: dict[str, str]) -> dict[str, Asset]:
        """A loop made before from the same drawing, prompt and seed is reused, when a try of it can be used;
        the rest are made and checked side by side, then stored here, on one thread, with their costs."""
        max_tries, jobs, out = self._settings.image_max_tries, [], {}
        for d in todo:
            prompt, still = loop_prompt(d), stills[d.id]
            base = int(hashlib.sha256("|".join([prompt, still.sha256]).encode()).hexdigest()[:8], 16) % 2**31
            keys = [hashlib.sha256("|".join([model, prompt, still.sha256, str(base + n), str(LOOP_SECONDS)]).encode())
                    .hexdigest() for n in range(max_tries)]
            cached = next((c for k in keys for c in self._assets.assets.list(cache_key=k) if c.review in USABLE), None)
            if cached is not None:
                logger.info("loop %s: reused %s", d.id, cached.id[:8])
                out[d.id] = self._assets.assets.add(Asset(
                    run_id=run_id, stage=STAGE, kind=AssetKind.VIDEO, path=cached.path, sha256=cached.sha256,
                    target_id=d.id, cache_key=cached.cache_key, provider=cached.provider, model=cached.model,
                    seed=cached.seed, prompt=cached.prompt, params={**cached.params, "reused_from": cached.id},
                    cost_usd=0.0, review=cached.review))
            else:
                jobs.append((d, prompt, still, base, keys))
        attempt = lambda job: self._tries(model, loop_brief(board, job[0], must_not_show(board, job[0], avoid)), *job)
        with ThreadPoolExecutor(max_workers=self._settings.video_workers) as pool:
            animated = list(pool.map(attempt, jobs))
        for (d, _, still, _, _), tries in zip(jobs, animated):
            out[d.id] = self._keep(run_id, d, still, tries)
        return {d.id: out[d.id] for d in todo}

    def _tries(self, model: str, brief: Brief, drawing: Drawing, prompt: str, still: Asset, base: int,
               keys: list[str]) -> list["Try"]:
        """Animate, cut out and check, again on a new seed until a try passes; a retry's prompt is given a
        correction written from what the last try's checks found. Runs beside other loops, so it only calls
        the models and reads frames; nothing is stored here."""
        url = asset_url(self._storage, self._video.upload, still)
        reference = [LabelledImage(label=f"reference: the drawing {drawing.id}", data=self._storage.get(still.path))]
        tries, correction, fix = [], None, None
        for n, key in enumerate(keys):
            asked = f"{prompt} {correction}" if correction else prompt
            reply = self._video.generate(model, asked, url, LOOP_SECONDS, seed=base + n, end_image_url=url)
            loop, cut, frame = self._cutter.cut(reply.data)
            made = Try(reply=reply, attempt=n + 1, cache_key=key, prompt=asked, correction=correction, fix=fix,
                       loop=loop, cut=cut, problems=loop.problems, notes=loop.problems)
            if cut is not None:                                 # a loop that is no cut-out needs no judge
                checklist, made.check = self._check.check(frame, reference, brief)
                made.problems = loop.problems + self._check.problems(checklist, brief)
                made.notes, made.score = loop.problems + self._check.notes(checklist), checklist.score
            tries.append(made)
            if not made.problems:
                break
            logger.warning("loop %s: try %d failed %d checks:\n%s", drawing.id, made.attempt, len(made.problems),
                           "\n".join(f"  - {p}" for p in made.problems))
            if n + 1 < len(keys):                               # what the loop must show is the brief's description
                correction, fix = write_correction(
                    self._llm, self._settings.image_check_model, logger, "loop", drawing.id, prompt,
                    brief.model_copy(update={"moment": brief.description}), made.problems)
        return tries

    def _keep(self, run_id: str, drawing: Drawing, still: Asset, tries: list["Try"]) -> Asset:
        """Store every try under its own file, with its transparent version beside it, and say which one the
        video uses: the try that passed; else the better one (one that closes and moves before one that does
        not, then the higher score, then the later try), as used_best; else none, when no try could be cut
        out, and the loop has failed."""
        assets = []
        for t in tries:
            asset_id = new_id()
            path = f"runs/{run_id}/loops/{drawing.id}/{t.attempt}-{asset_id[:8]}.mp4"
            self._storage.put(path, t.reply.data, t.reply.content_type)
            cut = path.removesuffix(".mp4") + "-cut.webm" if t.cut else None
            if cut:
                self._storage.put(cut, t.cut, "video/webm")
            assets.append(self._assets.assets.add(Asset(
                id=asset_id, run_id=run_id, stage=STAGE, kind=AssetKind.VIDEO, path=path,
                sha256=hashlib.sha256(t.reply.data).hexdigest(), target_id=drawing.id, cache_key=t.cache_key,
                provider=self._video.name, model=t.reply.model, seed=t.reply.seed, prompt=t.prompt,
                params={"try": t.attempt, "still": still.id, "seconds": t.loop.seconds, "fps": t.loop.fps,
                        "frames": t.loop.frames, "gap": t.loop.gap, "motion": t.loop.motion, "cut": cut,
                        "score": t.score, "correction": t.correction},
                cost_usd=t.reply.cost_usd, review_notes="\n".join(t.notes) or None,
                review=ReviewState.FAILED if t.problems else ReviewState.PASSED)))
            add_cost(self._assets, run_id, STAGE, self._video.name, t.reply.model, t.reply.cost_usd, drawing.id,
                     t.reply.seconds_requested, "second")
            for call in (c for c in (t.check, t.fix) if c is not None):
                add_llm_cost(self._assets, run_id, STAGE, call)
        if assets[-1].review == ReviewState.PASSED:
            logger.info("loop %s: passed on try %d", drawing.id, len(assets))
            return assets[-1]
        chosen, best = max(zip(tries, assets), key=lambda pair: (pair[0].cut is not None, not pair[0].loop.problems,
                                                                 pair[0].score, pair[0].attempt))
        if chosen.cut is not None:                              # it could be cut out, so it can be used
            best.review = ReviewState.USED_BEST
            self._assets.assets.update(best)
            logger.warning("loop %s: no try passed; using try %d, %d/10", drawing.id, chosen.attempt, chosen.score)
        else:
            logger.error("loop %s: no try could be cut out", drawing.id)
        return best


class Try(BaseModel):
    """One loop of a drawing with what the checks found; score 0 when it was no cut-out and never judged.
    prompt: what this try was made from; correction and fix: what a retry's prompt was given and the call
    that wrote it; cut: the transparent loop."""

    reply: VideoReply
    attempt: int
    cache_key: str
    prompt: str
    correction: str | None = None
    fix: LlmCall | None = None
    loop: LoopCheck
    cut: bytes | None
    problems: list[str]
    notes: list[str]
    score: int = 0
    check: LlmCall | None = None
