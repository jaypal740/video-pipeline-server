import hashlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel

from src.core.configs import Settings
from src.repository.asset.asset_repository import AssetRepository
from src.repository.run.run_repository import RunRepository
from src.schema.asset.asset_schema import Asset, ReviewState
from src.schema.asset.cost_entry_schema import CostEntry
from src.schema.base_schema import new_id, utcnow
from src.schema.enums import AssetKind, StageName, StageStatus
from src.schema.image.image_check_schema import Brief, Checklist, Correction
from src.schema.run.stage_state_schema import StageState
from src.schema.storyboard.storyboard_schema import CastMember, Place, Shot, ShotKind, Storyboard
from src.service.image.fal_image_service import RESOLUTION, FalImageService, ImageReply
from src.service.image.image_check_service import ImageCheckService
from src.service.lesson.lesson_rules import band_rules
from src.service.llm.llm_service import LabelledImage, LlmCall, LlmError, LlmService
from src.service.keyframe.keyframe_prompt import (CORRECTION, check_correction, correction_request, keyframe_brief,
                                                  keyframe_prompt, view_for, ref_label)
from src.service.sheet.sheet_prompt import PLATE, Style
from src.service.image.image_prices import image_model
from src.service.sheet.sheet_service import URL_SECONDS, USABLE, SheetService

PASSED = (ReviewState.PASSED, ReviewState.APPROVED)          # a try that met the bar; USABLE adds used_best
from src.service.storage.storage_service import Storage
from src.service.storyboard.storyboard_rules import HOST

logger = logging.getLogger(__name__)

ASPECT = "16:9"
# The stop-loss looks at the first this-many finished shots.
STOP_LOSS_AFTER = 8


class Attempt(BaseModel):
    """One draw and its check, made by a worker thread; the main thread turns it into Asset rows."""

    attempt: int
    seed: int
    thinking: str
    prompt: str
    correction: str | None = None
    fix: LlmCall | None = None
    score: int
    asked_for: list[dict]
    cache_key: str
    reply: ImageReply
    problems: list[str]
    check: LlmCall


class KeyframeService:
    """Stage 4b: one picture per shot, the frozen opening moment, drawn from start_state with the place
    plate and the subjects' sheets attached. Workers draw and check over the network; only the main thread
    writes rows. A shot whose sheets are not usable is left blocked until a person settles them."""

    def __init__(self, run_repository: RunRepository, asset_repository: AssetRepository, storage: Storage,
                 images: FalImageService, check: ImageCheckService, sheets: SheetService, llm: LlmService,
                 settings: Settings) -> None:
        self._runs = run_repository
        self._assets = asset_repository
        self._storage = storage
        self._images = images
        self._check = check
        self._llm = llm
        self._sheets = sheets
        self._settings = settings

    def run(self, run_id: str, force: bool = False) -> dict[str, Asset]:
        board = self._board(run_id)
        sheets = self._usable_sheets(run_id, board)
        style, model = Style.of(band_rules(board.grade_band)), self._settings.keyframe_model
        cast = {c.id: c for c in board.cast}
        places = {p.id: p for p in board.places}
        shots = [s for s in board.shots if s.repeat_of is None]
        plans, blocked = [], []
        for shot in shots:
            refs = self._refs_for(shot, cast, sheets)
            if refs is None:
                blocked.append(shot.id)
                continue
            subjects = [cast[s] for s in shot.subjects]
            prompt = keyframe_prompt(style, shot, places[shot.place], subjects)
            plans.append((shot, prompt, refs, keyframe_brief(shot, places[shot.place], subjects)))
        key = hashlib.sha256("|".join([model, *(f"{s.id}:{p}:{','.join(r.sha256 for r in refs)}"
                                               for s, p, refs, _ in plans)]).encode()).hexdigest()

        state = self._runs.stage(run_id, StageName.KEYFRAMES)
        if state is not None and not force and state.status == StageStatus.DONE and state.input_hash == key:
            logger.info("keyframes %s: reused, inputs unchanged", run_id[:8])
            return {s: self._assets.assets.get(a) for s, a in state.output["keyframes"].items()}
        state = state or self._runs.stage_states.add(StageState(run_id=run_id, stage=StageName.KEYFRAMES))
        state.status, state.attempts, state.error = StageStatus.RUNNING, state.attempts + 1, None
        state.input_hash, state.started_at, state.finished_at = key, utcnow(), None
        state = self._runs.stage_states.update(state)
        logger.info("keyframes %s: %s, %d shots to draw, %d blocked by sheets, %d repeats reused", run_id[:8], model,
                    len(plans), len(blocked), len(board.shots) - len(shots))
        self._errors: dict[str, str] = {}
        self._lock, self._finished, self._failed, self._stopped = threading.Lock(), 0, 0, None
        try:
            made = self._draw_all(run_id, model, plans)
        except Exception as exc:
            state.status, state.error, state.finished_at = StageStatus.FAILED, str(exc), utcnow()
            self._runs.stage_states.update(state)
            logger.error("keyframes %s: failed: %s", run_id[:8], exc)
            raise
        for shot in board.shots:                                    # a chorus repeat shows its original's picture
            if shot.repeat_of in made:
                made[shot.id] = made[shot.repeat_of]
        self._finish(run_id, state, made, blocked, len(plans))
        return made

    def _finish(self, run_id: str, state: StageState, made: dict[str, Asset], blocked: list[str], drawn: int) -> None:
        """No person decides here: a shot with a usable picture is done, any other shot fails the stage."""
        used_best = sorted({a.id for a in made.values() if a.review == ReviewState.USED_BEST})
        blocked = sorted(set(blocked) | set(self._errors))
        state.status = StageStatus.FAILED if blocked else StageStatus.DONE
        state.error = ("; ".join(f"{s}: {self._errors.get(s, 'no usable reference')}" for s in blocked) or None)
        state.output = {"keyframes": {s: a.id for s, a in made.items()}, "used_best": used_best, "blocked": blocked,
                        "errors": self._errors}
        state.finished_at = utcnow()
        self._runs.stage_states.update(state)
        retries = sum(a.params.get("try", 1) - 1 for s, a in made.items() if s == a.target_id)
        logger.log(logging.ERROR if blocked else logging.INFO,
                   "keyframes %s: %s, %d pictures, %d retries, %d used best, %d failed, $%.2f on the run so far, %.0fs",
                   run_id[:8], state.status, drawn, retries, len(used_best), len(blocked),
                   self._assets.run_cost(run_id), (state.finished_at - state.started_at).total_seconds())

    def recheck(self, run_id: str) -> dict[str, Asset]:
        """Judge every keyframe of this run that did not pass again, with the current brief; no image is drawn.
        The stage then takes the best passed try of each shot, or waits for a person as before."""
        board = self._board(run_id)
        state = self._runs.stage(run_id, StageName.KEYFRAMES)
        if state is None or not state.output:
            raise ValueError(f"Run {run_id} has no keyframes yet")
        cast = {c.id: c for c in board.cast}
        places = {p.id: p for p in board.places}
        shots = {s.id: s for s in board.shots}
        owner = {s: kept.run_id for s, a in state.output["keyframes"].items()      # the run that drew each shot
                 if (kept := self._assets.assets.get(a)) is not None}
        owner.update({s: run_id for s in state.output.get("blocked", [])})        # a failed shot's tries are ours
        tries = [t for shot_id, drawn_in in owner.items() if shot_id in shots and shots[shot_id].repeat_of is None
                 for t in self._assets.assets.list(run_id=drawn_in, stage=StageName.KEYFRAMES, target_id=shot_id)]
        self._errors = {k: v for k, v in state.output.get("errors", {}).items() if k not in shots
                        or shots[k].repeat_of is not None}
        for asset in tries:                                     # every try of each shot, in the run that drew it
            if asset.review in PASSED:
                continue
            shot = shots[asset.target_id]
            brief = keyframe_brief(shot, places[shot.place], [cast[s] for s in shot.subjects])
            refs = [self._assets.assets.get(r) for r in asset.params["refs"]]
            labelled = [LabelledImage(label=ref_label(kind, name), data=self._storage.get(r.path))
                        for (kind, name), r in zip([('place', shot.place), *(('cast member', s) for s in shot.subjects)], refs)]
            checklist, call = self._check.check(self._storage.get(asset.path), labelled, brief)
            problems = self._check.problems(checklist)
            self._cost(run_id, call.provider, call.model, call.cost_usd, shot.id, call.input_tokens + call.output_tokens,
                       "tokens")
            asset.params = {**asset.params, "score": checklist.score,
                            "asked_for": [a.model_dump() for a in checklist.asked_for]}
            asset.review = ReviewState.PASSED if not problems else ReviewState.FAILED
            asset.review_notes = "\n".join(problems) or None
            self._assets.assets.update(asset)
            logger.info("keyframe %s try %d rechecked: %s, %d/10", shot.id, asset.params["try"], asset.review,
                        checklist.score)
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

    def _keep(self, shot_id: str, tries: list[Asset]) -> Asset | None:
        """Which of a shot's tries the stage uses: the best passed one; else the best-scoring try at or above
        the floor, marked used_best; else none, and the shot is recorded as failed. Rows are updated to match."""
        passed = [a for a in tries if a.review in PASSED]
        if passed:
            return max(passed, key=lambda a: a.params.get("score") or 0)
        best = max(tries, key=lambda a: a.params.get("score") or 0)
        for a in tries:
            review = (ReviewState.USED_BEST if a is best and (a.params.get("score") or 0) >= self._settings.image_floor_score
                      else ReviewState.FAILED)
            if a.review != review:
                a.review = a.review if a.review == ReviewState.APPROVED else review
                self._assets.assets.update(a)
        if best.review == ReviewState.USED_BEST:
            logger.warning("keyframe %s: no try passed; using the best, %s/10: %s", shot_id, best.params.get("score"),
                           (best.review_notes or "").splitlines()[0] if best.review_notes else "")
            return best
        self._errors[shot_id] = (f"failed every draw; best {best.params.get('score')}/10: "
                                 + "; ".join((best.review_notes or "").splitlines()))
        logger.error("keyframe %s: %s", shot_id, self._errors[shot_id])
        return None

    # --- drawing ------------------------------------------------------------------------------------------

    def _draw_all(self, run_id: str, model: str, plans: list) -> dict[str, Asset]:
        made: dict[str, Asset] = {}
        todo = []
        for shot, prompt, refs, brief in plans:
            seeds = self._tries(model, prompt, refs, self._base_seed(prompt, refs))
            cached = {key: c for *_, key in seeds if (c := self._assets.find_cached(key)) is not None}
            tried = list(cached.values())                       # a seed is never drawn twice: same seed, same picture
            usable = next((c for c in tried if c.review in USABLE), None)
            left = [s for s in seeds if s[3] not in cached]
            if usable is not None:                              # the run that drew it keeps it; no new row
                logger.info("keyframe %s: reused %s (%s)", shot.id, usable.id[:8], usable.review)
                made[shot.id] = usable
            elif not left:                                      # every seed tried, none good enough
                self._keep(shot.id, tried)
            else:                                               # the last try drawn so far says what to correct
                last = max(tried, key=lambda c: c.params.get("try", 1), default=None)
                findings = (last.review_notes or "").splitlines() if last is not None else []
                todo.append((shot, prompt, refs, brief, left, findings))
        if not todo:
            return made
        self._guard_spend(run_id, model, len(todo))
        ref_urls = {r.id: self._url(r) for _, _, refs, *_ in todo for r in refs}
        ref_bytes = {r.id: self._storage.get(r.path) for _, _, refs, *_ in todo for r in refs}
        with ThreadPoolExecutor(max_workers=self._settings.image_workers) as pool:
            futures = {pool.submit(self._attempts, model, ref_urls, ref_bytes, *job): job for job in todo}
            for future in as_completed(futures):               # stored the moment each shot finishes
                shot, prompt, refs, *_ = futures[future]
                try:
                    tries = future.result()
                except Exception as exc:                        # one shot's error never loses the others
                    logger.error("keyframe %s: error, left for review: %s", shot.id, exc)
                    self._errors[shot.id] = str(exc)
                    continue
                if not tries:                                   # skipped: the stop-loss had fired
                    self._errors[shot.id] = self._stopped
                    continue
                kept = self._store_attempts(run_id, shot, refs, tries)
                if kept is not None:
                    made[shot.id] = kept
        if self._stopped:
            logger.error("keyframes %s: %s; %d shots not drawn", run_id[:8], self._stopped,
                         sum(1 for e in self._errors.values() if e == self._stopped))
        return made

    def _guard_spend(self, run_id: str, model: str, count: int) -> None:
        """Log what this drawing will cost and refuse to start if the run could pass the spend cap."""
        spec = image_model(model)
        each = spec.prices[ASPECT if spec.family == "gpt_image" else RESOLUTION] + spec.per_reference * 3
        worst = count * self._settings.image_max_tries * (each + 0.002)
        spent = self._assets.run_cost(run_id)
        logger.info("keyframes %s: %d shots x $%.3f, about $%.2f first time, at most $%.2f with retries; $%.2f spent "
                    "on the run so far, cap $%.2f", run_id[:8], count, each, count * (each + 0.002), worst, spent,
                    self._settings.spend_cap_usd)
        if spent + worst > self._settings.spend_cap_usd:
            raise ValueError(f"drawing {count} keyframes could cost ${worst:.2f} on top of ${spent:.2f} already spent, "
                             f"past the cap of ${self._settings.spend_cap_usd:.2f}; raise PIPELINE_SPEND_CAP_USD "
                             "or reduce the run")

    def _attempts(self, model: str, ref_urls: dict, ref_bytes: dict, shot: Shot, prompt: str, refs: list[Asset],
                  brief: Brief, seeds: list[tuple[int, int, str, str]], findings: list[str]) -> list[Attempt]:
        """Network only: draw, check, retry on a new seed with a correction written from the last try's
        findings. No database access in here. The workers keep the stop-loss count themselves, so no draw
        starts once it has fired, whatever the timing."""
        if self._stopped:
            return []
        urls = [ref_urls[r.id] for r in refs]
        labelled = [LabelledImage(label=ref_label(kind, name), data=ref_bytes[r.id])
                    for (kind, name), r in zip([('place', shot.place), *(('cast member', s) for s in shot.subjects)], refs)]
        tries = []
        for attempt, seed, thinking, cache_key in seeds:
            correction, fix = self._correction(shot, prompt, brief, findings) if findings else (None, None)
            drawn = f"{prompt} {correction}" if correction else prompt
            reply = self._images.generate(model, drawn, urls, ASPECT, seed=seed, thinking=thinking)
            checklist, call = self._check.check(reply.data, labelled, brief)
            problems = findings = self._check.problems(checklist)
            tries.append(Attempt(attempt=attempt, seed=seed, thinking=thinking, prompt=drawn, correction=correction,
                                 fix=fix, score=checklist.score,
                                 asked_for=[a.model_dump() for a in checklist.asked_for],
                                 cache_key=cache_key, reply=reply,
                                 problems=problems, check=call))
            if not problems:
                logger.info("keyframe %s: passed on try %d, $%.3f, %.1fs", shot.id, attempt, reply.cost_usd,
                            reply.seconds)
                self._count(passed=True)
                return tries
            logger.warning("keyframe %s: try %d failed %d checks:\n%s", shot.id, attempt, len(problems),
                           "\n".join(f"  - {p}" for p in problems))
        logger.warning("keyframe %s: left for review after %d tries", shot.id, len(seeds))
        self._count(passed=False)
        return tries

    def _correction(self, shot: Shot, prompt: str, brief: Brief, findings: list[str]) -> tuple[str | None, LlmCall | None]:
        """A short correction for the retry, from the failed prompt, the opening moment and the judge's
        findings; when none can be written the retry is drawn from the plain prompt."""
        try:
            fix, call = self._llm.parse(self._settings.image_check_model, CORRECTION,
                                        correction_request(prompt, brief.moment or "", findings), Correction,
                                        check=check_correction, effort="low")
        except LlmError as exc:
            logger.warning("keyframe %s: no correction, retrying the plain prompt: %s", shot.id, exc)
            return None, exc.call
        logger.info("keyframe %s: retry corrected: %s", shot.id, fix.text)
        return fix.text, call

    def _count(self, passed: bool) -> None:
        """The stop-loss: after the first STOP_LOSS_AFTER finished shots, too many failures pause the stage."""
        with self._lock:
            self._finished += 1
            self._failed += not passed
            if self._finished == STOP_LOSS_AFTER and self._failed > STOP_LOSS_AFTER * self._settings.image_stop_loss \
                    and not self._stopped:
                self._stopped = (f"stopped: {self._failed} of the first {self._finished} shots failed every draw; "
                                 "fix the cause before drawing more")

    def _store_attempts(self, run_id: str, shot: Shot, refs: list[Asset], tries: list[Attempt]) -> Asset | None:
        stored = []
        for t in tries:
            asset_id = new_id()
            path = f"runs/{run_id}/keyframes/{shot.id}/{t.attempt}-{asset_id[:8]}.png"
            self._storage.put(path, t.reply.data, t.reply.content_type)
            review = ReviewState.PASSED if not t.problems else ReviewState.FAILED
            asset = self._assets.assets.add(Asset(
                id=asset_id, run_id=run_id, stage=StageName.KEYFRAMES, kind=AssetKind.IMAGE, path=path,
                sha256=hashlib.sha256(t.reply.data).hexdigest(), target_id=shot.id, cache_key=t.cache_key,
                provider=self._images.name, model=t.reply.model, seed=t.reply.seed, prompt=t.prompt,
                params={"view": "keyframe", "try": t.attempt, "aspect": ASPECT, "resolution": RESOLUTION,
                        "thinking": t.thinking, "refs": [r.id for r in refs], "seconds": t.reply.seconds,
                        "score": t.score, "asked_for": t.asked_for, "correction": t.correction},
                cost_usd=t.reply.cost_usd, review=review, review_notes="\n".join(t.problems) or None))
            self._cost(run_id, self._images.name, t.reply.model, t.reply.cost_usd, shot.id, 1, "image")
            for call in (t.check, t.fix) if t.fix is not None else (t.check,):
                self._cost(run_id, call.provider, call.model, call.cost_usd, shot.id,
                           call.input_tokens + call.output_tokens, "tokens")
            stored.append(asset)
        return self._keep(shot.id, stored)

    # --- what each shot needs --------------------------------------------------------------------------------

    def _refs_for(self, shot: Shot, cast: dict[str, CastMember], sheets: dict[str, dict[str, Asset]]) -> list[Asset] | None:
        """The place plate first, then one view per subject; None when any of them is not usable yet."""
        plate = sheets.get(shot.place, {}).get(PLATE)
        if plate is None:
            return None
        refs = [plate]
        for subject in shot.subjects:
            view = view_for(cast[subject], shot)
            key = f"{HOST}@{shot.place}" if subject == HOST else subject
            sheet = sheets.get(key, {}).get(view)
            if sheet is None:
                return None
            refs.append(sheet)
        return refs

    def _usable_sheets(self, run_id: str, board: Storyboard) -> dict[str, dict[str, Asset]]:
        """Passed or approved sheets for this run, plus the band's host images in this video's outfit kit."""
        state = self._runs.stage(run_id, StageName.VISUAL_IDENTITY)
        if state is None or state.status != StageStatus.DONE:
            raise ValueError(f"Run {run_id} has no finished reference sheets (sheets stage: "
                             f"{state.status if state else 'not run'}); run the sheets stage first")
        found: dict[str, dict[str, Asset]] = {}
        for target, views in state.output["sheets"].items():
            for view, asset_id in views.items():
                asset = self._assets.assets.get(asset_id)
                if asset is not None and asset.review in USABLE:
                    found.setdefault(target, {})[view] = asset
        for place in board.places:                                  # the host as dressed for each place
            if place.host_outfit:
                key = self._sheets.outfit_key(board.grade_band, place.host_outfit)
                found[f"{HOST}@{place.id}"] = self._sheets.host_sheets(board.grade_band, key)
        return found

    # --- small helpers ------------------------------------------------------------------------------------

    def _url(self, asset: Asset) -> str:
        return self._storage.url(asset.path, URL_SECONDS) or self._images.upload(self._storage.get(asset.path),
                                                                                 "image/png")

    @staticmethod
    def _base_seed(prompt: str, refs: list[Asset]) -> int:
        return int(hashlib.sha256("|".join([prompt, *(r.sha256 for r in refs)]).encode()).hexdigest()[:8], 16) % 2**31

    def _tries(self, model: str, prompt: str, refs: list[Asset], base: int) -> list[tuple[int, int, str, str]]:
        """A shot's tries in order: try number, seed, thinking level (the last try thinks harder) and cache key.
        Shared by the cache lookup and the drawing so they always agree; a try keeps its number even when
        it is the only one left to draw. The key uses the shot's own prompt, not a retry's correction, so a
        corrected retry is still found and never drawn twice."""
        max_tries = self._settings.image_max_tries
        return [(i + 1, base + i, "high" if i == max_tries - 1 else "medium",
                 self._cache_key(model, prompt, refs, base + i, "high" if i == max_tries - 1 else "medium"))
                for i in range(max_tries)]

    @staticmethod
    def _cache_key(model: str, prompt: str, refs: list[Asset], seed: int, thinking: str) -> str:
        parts = [model, prompt, *(r.sha256 for r in refs), str(seed), ASPECT, RESOLUTION, thinking]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    def _cost(self, run_id: str, provider: str, model: str, cost: float, target_id: str, units: float, unit: str) -> None:
        self._assets.cost_entries.add(CostEntry(run_id=run_id, stage=StageName.KEYFRAMES, provider=provider, model=model,
                                                cost_usd=cost, target_id=target_id, units=units, unit=unit))

    def _board(self, run_id: str) -> Storyboard:
        state = self._runs.stage(run_id, StageName.STORYBOARD)
        if state is None or state.status != StageStatus.DONE:
            raise ValueError(f"Run {run_id} has no storyboard; run the storyboard stage first")
        return Storyboard(**state.output)
