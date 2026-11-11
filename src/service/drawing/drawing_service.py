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
from src.schema.storyboard.script_schema import AnimationScript, Drawing, DrawingKind
from src.service.drawing.cutout import cut_out
from src.service.drawing.drawing_prompt import aspect_of, drawing_brief, drawing_prompt, must_not_show, reference_of
from src.service.image.fal_image_service import RESOLUTION, FalImageService, ImageReply
from src.service.image.image_check_service import ImageCheckService
from src.service.llm.llm_service import LabelledImage, LlmCall, LlmService
from src.service.stage.stage_common import add_cost, add_llm_cost, asset_url, fail_stage, script_of, write_correction
from src.service.storage.storage_service import Storage

logger = logging.getLogger(__name__)

STAGE = StageName.VISUAL_IDENTITY
USABLE = (ReviewState.PASSED, ReviewState.APPROVED, ReviewState.USED_BEST)


class DrawingService:
    """Stage 4: every drawing of the animation script, drawn once. A backdrop fills the frame; everything
    else is drawn alone on the key colour and cut out by code. The performer's lead drawing is drawn first
    and attached to its other drawings, and a pose is drawn from the drawing it is a pose of, so one thing
    looks the same everywhere. Each drawing is checked by code as a cut-out and by the judge against its
    look. One that fails is drawn again on a new seed, told what the check found wrong; when that fails too
    the better-scoring try is used (used_best). Only a drawing no try of which could be cut out fails the
    stage. A re-run with the same script reuses the stored drawings."""

    def __init__(self, run_repository: RunRepository, asset_repository: AssetRepository, storage: Storage,
                 images: FalImageService, check: ImageCheckService, llm: LlmService, settings: Settings) -> None:
        self._runs = run_repository
        self._assets = asset_repository
        self._storage = storage
        self._images = images
        self._check = check
        self._llm = llm
        self._settings = settings

    def run(self, run_id: str, until: float | None = None, force: bool = False) -> dict[str, Asset]:
        """until: draw only what the beats starting before that second of the song need."""
        if self._runs.runs.get(run_id) is None:
            raise LookupError(f"Run {run_id} not found")
        board = script_of(self._runs, run_id)
        avoid = {b.id: b.avoid for b in LessonPlan(**self._runs.stage(run_id, StageName.LESSON).output).beats}
        todo = _wanted(board, until)
        refs = {d.id: reference_of(board, d) for d in todo}
        model = self._settings.drawing_model
        key = hashlib.sha256("|".join([model] + [
            f"{d.id}:{refs[d.id].id if refs[d.id] else ''}:{drawing_prompt(board, d, refs[d.id])}" for d in todo
        ]).encode()).hexdigest()

        state = self._runs.stage(run_id, STAGE)
        if state is not None and not force and state.status == StageStatus.DONE and state.input_hash == key:
            logger.info("drawings %s: reused, inputs unchanged", run_id[:8])
            return {name: self._assets.assets.get(a) for name, a in state.output["drawings"].items()}
        state = state or self._runs.stage_states.add(StageState(run_id=run_id, stage=STAGE))
        state.status, state.attempts, state.error = StageStatus.RUNNING, state.attempts + 1, None
        state.input_hash, state.started_at, state.finished_at = key, utcnow(), None
        state = self._runs.stage_states.update(state)
        logger.info("drawings %s: %s, %d of %d drawings to draw", run_id[:8], model, len(todo), len(board.drawings))
        made: dict[str, Asset] = {}
        try:
            for level in _levels(todo, refs):
                assets = self._draw_all(run_id, model, board, level, refs, made, avoid)
                made.update(zip((d.id for d in level), assets))
                if any(a.review == ReviewState.FAILED for a in assets):     # nothing is drawn from a failed drawing
                    break
        except Exception as exc:
            fail_stage(self._runs, state, exc, logger, "drawings")
            raise
        failed = [name for name, a in made.items() if a.review == ReviewState.FAILED]
        state.status = StageStatus.FAILED if failed else StageStatus.DONE
        state.error = "; ".join(f"{name}: failed every draw: {made[name].review_notes}" for name in failed) or None
        state.output = {"drawings": {name: a.id for name, a in made.items()}, "until": until,
                        "used_best": [name for name, a in made.items() if a.review == ReviewState.USED_BEST],
                        "not_drawn": [d.id for d in todo if d.id not in made]}
        state.finished_at = utcnow()
        self._runs.stage_states.update(state)
        logger.log(logging.ERROR if failed else logging.INFO,
                   "drawings %s: %s, %d drawn, %d retries, %d used best, %d failed, $%.2f on the run so far, %.0fs",
                   run_id[:8], state.status, len(made), sum(a.params.get("try", 1) - 1 for a in made.values()),
                   len(state.output["used_best"]), len(failed), self._assets.run_cost(run_id),
                   (state.finished_at - state.started_at).total_seconds())
        return made

    def _draw_all(self, run_id: str, model: str, board: AnimationScript, level: list[Drawing],
                  refs: dict[str, Drawing | None], made: dict[str, Asset], avoid: dict[str, str]) -> list[Asset]:
        """One round of drawings. A picture drawn before with the same prompt, reference and seed is reused,
        when a try of it passed and it still cuts out; the rest are drawn and checked side by side, then
        stored here, on one thread, with their costs."""
        max_tries, jobs, out = self._settings.image_max_tries, [], {}
        for d in level:
            ref = [made[refs[d.id].id]] if refs[d.id] else []
            prompt, aspect = drawing_prompt(board, d, refs[d.id]), aspect_of(d)
            base = int(hashlib.sha256("|".join([prompt, *(r.sha256 for r in ref)]).encode()).hexdigest()[:8], 16) % 2**31
            keys = [hashlib.sha256("|".join([model, prompt, *(r.sha256 for r in ref), str(base + n), aspect, RESOLUTION,
                                             _thinking(n + 1, max_tries)]).encode()).hexdigest() for n in range(max_tries)]
            cached = next((c for k in keys for c in self._assets.assets.list(cache_key=k) if c.review in USABLE), None)
            again = cut_out(self._storage.get(cached.path)) if cached is not None and cached.params.get("cut") else None
            if cached is not None and not (again and again[1]):
                logger.info("drawing %s: reused %s", d.id, cached.id[:8])
                if again:                                       # cut out again, in case the cutting has changed
                    self._storage.put(cached.params["cut"], again[0], "image/png")
                out[d.id] = self._assets.assets.add(Asset(
                    run_id=run_id, stage=STAGE, kind=AssetKind.IMAGE, path=cached.path, sha256=cached.sha256,
                    target_id=d.id, cache_key=cached.cache_key, provider=cached.provider, model=cached.model,
                    seed=cached.seed, prompt=cached.prompt, params={**cached.params, "reused_from": cached.id},
                    cost_usd=0.0, review=ReviewState.PASSED))
            else:
                jobs.append((d, prompt, aspect, ref, base, keys))
        attempt = lambda job: self._tries(model, drawing_brief(board, job[0], must_not_show(board, job[0], avoid)), *job)
        with ThreadPoolExecutor(max_workers=self._settings.image_workers) as pool:
            drawn = list(pool.map(attempt, jobs))
        for (d, _, aspect, ref, _, _), tries in zip(jobs, drawn):
            out[d.id] = self._keep(run_id, d, aspect, ref, tries)
        return [out[d.id] for d in level]

    def _tries(self, model: str, brief: Brief, drawing: Drawing, prompt: str, aspect: str,
               ref: list[Asset], base: int, keys: list[str]) -> list["Try"]:
        """Draw, cut out and check, again on a new seed until a try passes; a retry's prompt is given a
        correction written from what the last try's checks found. Runs beside other drawings, so it only
        calls the models; nothing is stored here."""
        ref_urls = [asset_url(self._storage, self._images.upload, r) for r in ref]
        labelled = [LabelledImage(label=f"reference: the drawing {r.target_id}", data=self._storage.get(r.path))
                    for r in ref if drawing.of]                 # the judge compares a pose only with what it is a pose of
        tries, correction, fix = [], None, None
        for n, key in enumerate(keys):
            drawn = f"{prompt} {correction}" if correction else prompt
            reply = self._images.generate(model, drawn, ref_urls, aspect, seed=base + n,
                                          thinking=_thinking(n + 1, len(keys)))
            cut, problems = (None, []) if drawing.kind == DrawingKind.BACKGROUND else cut_out(reply.data)
            made = Try(reply=reply, attempt=n + 1, cache_key=key, prompt=drawn, correction=correction, fix=fix,
                       cut=cut, problems=problems, notes=problems)
            if not problems:                                    # a picture that is no cut-out needs no judge
                checklist, made.check = self._check.check(reply.data, labelled, brief)
                made.problems, made.notes = self._check.problems(checklist, brief), self._check.notes(checklist)
                made.score = checklist.score
            tries.append(made)
            if not made.problems:
                break
            logger.warning("drawing %s: try %d failed %d checks:\n%s", drawing.id, made.attempt, len(made.problems),
                           "\n".join(f"  - {p}" for p in made.problems))
            if n + 1 < len(keys):                               # what the picture must show is the brief's description
                correction, fix = write_correction(
                    self._llm, self._settings.image_check_model, logger, "drawing", drawing.id, prompt,
                    brief.model_copy(update={"moment": brief.description}), made.problems)
        return tries

    def _keep(self, run_id: str, drawing: Drawing, aspect: str, ref: list[Asset], tries: list["Try"]) -> Asset:
        """Store every try under its own file, with its cut-out beside it, and say which one the video uses:
        the try that passed; else the better-scoring one, the later on a tie (used_best); else none, when no
        try could be cut out, and the drawing has failed."""
        assets = []
        for t in tries:
            asset_id = new_id()
            path = f"runs/{run_id}/drawings/{drawing.id}/{t.attempt}-{asset_id[:8]}.png"
            self._storage.put(path, t.reply.data, t.reply.content_type)
            cut = path.removesuffix(".png") + "-cut.png" if t.cut else None
            if cut:
                self._storage.put(cut, t.cut, "image/png")
            assets.append(self._assets.assets.add(Asset(
                id=asset_id, run_id=run_id, stage=STAGE, kind=AssetKind.IMAGE, path=path,
                sha256=hashlib.sha256(t.reply.data).hexdigest(), target_id=drawing.id, cache_key=t.cache_key,
                provider=self._images.name, model=t.reply.model, seed=t.reply.seed, prompt=t.prompt,
                params={"try": t.attempt, "aspect": aspect, "resolution": RESOLUTION, "refs": [r.id for r in ref],
                        "seconds": t.reply.seconds, "cut": cut, "score": t.score, "correction": t.correction},
                cost_usd=t.reply.cost_usd, review_notes="\n".join(t.notes) or None,
                review=ReviewState.FAILED if t.problems else ReviewState.PASSED)))
            add_cost(self._assets, run_id, STAGE, self._images.name, t.reply.model, t.reply.cost_usd, drawing.id, 1,
                     "image")
            if t.check:
                add_cost(self._assets, run_id, STAGE, t.check.provider, t.check.model, t.check.cost_usd, drawing.id,
                         t.check.input_tokens + t.check.output_tokens, "tokens")
            if t.fix:
                add_llm_cost(self._assets, run_id, STAGE, t.fix)
        if assets[-1].review == ReviewState.PASSED:
            logger.info("drawing %s: passed on try %d", drawing.id, len(assets))
            return assets[-1]
        judged, best = max(zip(tries, assets), key=lambda pair: (pair[0].check is not None, pair[0].score, pair[0].attempt))
        if judged.check is not None:                            # it could be cut out, so it can be used
            best.review = ReviewState.USED_BEST
            self._assets.assets.update(best)
            logger.warning("drawing %s: no try passed; using the best, %d/10", drawing.id, best.params["score"])
        else:
            logger.error("drawing %s: failed every draw; best %d/10", drawing.id, best.params["score"])
        return best


class Try(BaseModel):
    """One draw of a drawing with what the checks found; score 0 when it was no cut-out and never judged.
    prompt: what this try was drawn from; correction and fix: what a retry's prompt was given and the call
    that wrote it."""

    reply: ImageReply
    attempt: int
    cache_key: str
    prompt: str
    correction: str | None = None
    fix: LlmCall | None = None
    cut: bytes | None
    problems: list[str]
    notes: list[str]
    score: int = 0
    check: LlmCall | None = None


def _thinking(attempt: int, max_tries: int) -> str:
    """Plain drawings want minimal thinking; the last try of a drawing thinks a little more."""
    return "medium" if attempt == max_tries else "minimal"


def _wanted(board: AnimationScript, until: float | None) -> list[Drawing]:
    """The drawings the beats starting before `until` use, with the drawings those are drawn from."""
    used = {i for b in board.beats if until is None or b.start < until
            for i in (b.background, *(x for a in b.actions for x in (a.drawing, a.to)))}
    while True:
        more = {r.id for d in board.drawings if d.id in used and (r := reference_of(board, d))} - used
        if not more:
            return [d for d in board.drawings if d.id in used]
        used |= more


def _levels(todo: list[Drawing], refs: dict[str, Drawing | None]) -> list[list[Drawing]]:
    """The drawings in rounds: first those drawn from words alone, then those drawn from a finished one."""
    levels, done = [], set()
    while len(done) < len(todo):
        level = [d for d in todo if d.id not in done and (refs[d.id] is None or refs[d.id].id in done)]
        if not level:
            raise ValueError("drawings are drawn from each other in a circle: "
                             + ", ".join(d.id for d in todo if d.id not in done))
        levels.append(level)
        done |= {d.id for d in level}
    return levels
