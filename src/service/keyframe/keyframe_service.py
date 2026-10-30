import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from src.core.configs import Settings
from src.repository.asset.asset_repository import AssetRepository
from src.repository.run.run_repository import RunRepository
from src.schema.asset.asset_schema import Asset, ReviewState
from src.schema.asset.cost_entry_schema import CostEntry
from src.schema.base_schema import new_id, utcnow
from src.schema.enums import AssetKind, StageName, StageStatus
from src.schema.image.image_check_schema import Checklist
from src.schema.run.stage_state_schema import StageState
from src.schema.storyboard.storyboard_schema import CastMember, Place, Shot, ShotKind, Storyboard
from src.service.image.fal_image_service import RESOLUTION, FalImageService, ImageReply
from src.service.image.image_check_service import ImageCheckService
from src.service.lesson.lesson_rules import band_rules
from src.service.llm.llm_service import LabelledImage, LlmCall
from src.service.keyframe.keyframe_prompt import keyframe_prompt, keyframe_questions, view_for
from src.service.sheet.sheet_prompt import PLATE, Style
from src.service.sheet.sheet_service import MAX_TRIES, URL_SECONDS, USABLE, SheetService
from src.service.storage.storage_service import Storage
from src.service.storyboard.storyboard_rules import HOST

logger = logging.getLogger(__name__)

ASPECT = "16:9"


class Attempt(BaseModel):
    """One draw and its check, made by a worker thread; the main thread turns it into Asset rows."""

    attempt: int
    seed: int
    thinking: str
    cache_key: str
    reply: ImageReply
    problems: list[str]
    check: LlmCall


class KeyframeService:
    """Stage 4b: one picture per shot, the frozen opening moment, drawn from start_state with the place
    plate and the subjects' sheets attached. Workers draw and check over the network; only the main thread
    writes rows. A shot whose sheets are not usable is left blocked until a person settles them."""

    def __init__(self, run_repository: RunRepository, asset_repository: AssetRepository, storage: Storage,
                 images: FalImageService, check: ImageCheckService, sheets: SheetService, settings: Settings) -> None:
        self._runs = run_repository
        self._assets = asset_repository
        self._storage = storage
        self._images = images
        self._check = check
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
            plans.append((shot, prompt, refs, keyframe_questions(shot, places[shot.place], subjects)))
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
        waiting = sorted({a.id for a in made.values() if a.review == ReviewState.NEEDS_REVIEW})
        state.status = StageStatus.WAITING_GATE if waiting or blocked else StageStatus.DONE
        state.output = {"keyframes": {s: a.id for s, a in made.items()}, "needs_review": waiting, "blocked": blocked}
        state.finished_at = utcnow()
        self._runs.stage_states.update(state)
        retries = sum(a.params.get("try", 1) - 1 for s, a in made.items() if s == a.target_id)
        logger.info("keyframes %s: %s, %d pictures, %d retries, %d need review, %d blocked, $%.2f on the run so far, "
                    "%.0fs", run_id[:8], state.status, len(plans), retries, len(waiting), len(blocked),
                    self._assets.run_cost(run_id), (state.finished_at - state.started_at).total_seconds())
        return made

    # --- drawing ------------------------------------------------------------------------------------------

    def _draw_all(self, run_id: str, model: str, plans: list) -> dict[str, Asset]:
        made: dict[str, Asset] = {}
        todo = []
        for shot, prompt, refs, questions in plans:
            base = self._base_seed(prompt, refs)
            cached = self._assets.find_cached(self._cache_key(model, prompt, refs, base, "medium"))
            if cached is not None and cached.review in USABLE:
                logger.info("keyframe %s: reused %s", shot.id, cached.id[:8])
                made[shot.id] = self._reuse(run_id, shot, cached, refs)
            else:
                todo.append((shot, prompt, refs, questions, base))
        if not todo:
            return made
        ref_urls = {r.id: self._url(r) for _, _, refs, _, _ in todo for r in refs}
        ref_bytes = {r.id: self._storage.get(r.path) for _, _, refs, _, _ in todo for r in refs}
        with ThreadPoolExecutor(max_workers=self._settings.image_workers) as pool:
            attempts = list(pool.map(lambda job: self._attempts(model, ref_urls, ref_bytes, *job), todo))
        for (shot, prompt, refs, _, _), tries in zip(todo, attempts):
            made[shot.id] = self._store_attempts(run_id, shot, prompt, refs, tries)
        return made

    def _attempts(self, model: str, ref_urls: dict, ref_bytes: dict, shot: Shot, prompt: str, refs: list[Asset],
                  questions: list[str], base: int) -> list[Attempt]:
        """Network only: draw, check, retry on a new seed. No database access in here."""
        urls = [ref_urls[r.id] for r in refs]
        labelled = [LabelledImage(label=f"reference for {r.target_id}", data=ref_bytes[r.id]) for r in refs]
        tries = []
        for attempt in range(1, MAX_TRIES + 1):
            seed, thinking = base + attempt - 1, "high" if attempt == MAX_TRIES else "medium"
            reply = self._images.generate(model, prompt, urls, ASPECT, seed=seed, thinking=thinking)
            checklist, call = self._check.check(reply.data, labelled, questions)
            problems = checklist.problems()
            tries.append(Attempt(attempt=attempt, seed=seed, thinking=thinking,
                                 cache_key=self._cache_key(model, prompt, refs, seed, thinking), reply=reply,
                                 problems=problems, check=call))
            if not problems:
                logger.info("keyframe %s: passed on try %d, $%.3f, %.1fs", shot.id, attempt, reply.cost_usd,
                            reply.seconds)
                return tries
            logger.warning("keyframe %s: try %d failed %d checks:\n%s", shot.id, attempt, len(problems),
                           "\n".join(f"  - {p}" for p in problems))
        logger.warning("keyframe %s: left for review after %d tries", shot.id, MAX_TRIES)
        return tries

    def _store_attempts(self, run_id: str, shot: Shot, prompt: str, refs: list[Asset], tries: list[Attempt]) -> Asset:
        final = None
        for t in tries:
            asset_id = new_id()
            path = f"runs/{run_id}/keyframes/{shot.id}/{t.attempt}-{asset_id[:8]}.png"
            self._storage.put(path, t.reply.data, t.reply.content_type)
            review = (ReviewState.PASSED if not t.problems else
                      ReviewState.NEEDS_REVIEW if t.attempt == MAX_TRIES else ReviewState.FAILED)
            asset = self._assets.assets.add(Asset(
                id=asset_id, run_id=run_id, stage=StageName.KEYFRAMES, kind=AssetKind.IMAGE, path=path,
                sha256=hashlib.sha256(t.reply.data).hexdigest(), target_id=shot.id, cache_key=t.cache_key,
                provider=self._images.name, model=t.reply.model, seed=t.reply.seed, prompt=prompt,
                params={"view": "keyframe", "try": t.attempt, "aspect": ASPECT, "resolution": RESOLUTION,
                        "thinking": t.thinking, "refs": [r.id for r in refs], "seconds": t.reply.seconds},
                cost_usd=t.reply.cost_usd, review=review, review_notes="\n".join(t.problems) or None))
            self._cost(run_id, self._images.name, t.reply.model, t.reply.cost_usd, shot.id, 1, "image")
            self._cost(run_id, t.check.provider, t.check.model, t.check.cost_usd, shot.id,
                       t.check.input_tokens + t.check.output_tokens, "tokens")
            final = asset
        return final

    def _reuse(self, run_id: str, shot: Shot, cached: Asset, refs: list[Asset]) -> Asset:
        return self._assets.assets.add(Asset(
            run_id=run_id, stage=StageName.KEYFRAMES, kind=AssetKind.IMAGE, path=cached.path, sha256=cached.sha256,
            target_id=shot.id, cache_key=cached.cache_key, provider=cached.provider, model=cached.model,
            seed=cached.seed, prompt=cached.prompt, params={**cached.params, "refs": [r.id for r in refs],
                                                            "reused_from": cached.id},
            cost_usd=0.0, review=ReviewState.PASSED))

    # --- what each shot needs --------------------------------------------------------------------------------

    def _refs_for(self, shot: Shot, cast: dict[str, CastMember], sheets: dict[str, dict[str, Asset]]) -> list[Asset] | None:
        """The place plate first, then one view per subject; None when any of them is not usable yet."""
        plate = sheets.get(shot.place, {}).get(PLATE)
        if plate is None:
            return None
        refs = [plate]
        for subject in shot.subjects:
            view = view_for(cast[subject], shot)
            sheet = sheets.get(subject, {}).get(view)
            if sheet is None:
                return None
            refs.append(sheet)
        return refs

    def _usable_sheets(self, run_id: str, board: Storyboard) -> dict[str, dict[str, Asset]]:
        """Passed or approved sheets for this run, plus the band's host images in this video's outfit kit."""
        state = self._runs.stage(run_id, StageName.VISUAL_IDENTITY)
        if state is None or state.status not in (StageStatus.DONE, StageStatus.WAITING_GATE):
            raise ValueError(f"Run {run_id} has no reference sheets; run the sheets stage first")
        found: dict[str, dict[str, Asset]] = {}
        for target, views in state.output["sheets"].items():
            for view, asset_id in views.items():
                asset = self._assets.assets.get(asset_id)
                if asset is not None and asset.review in USABLE:
                    found.setdefault(target, {})[view] = asset
        found[HOST] = self._sheets.host_sheets(board.grade_band, board.host_outfit)
        return found

    # --- small helpers ------------------------------------------------------------------------------------

    def _url(self, asset: Asset) -> str:
        return self._storage.url(asset.path, URL_SECONDS) or self._images.upload(self._storage.get(asset.path),
                                                                                 "image/png")

    @staticmethod
    def _base_seed(prompt: str, refs: list[Asset]) -> int:
        return int(hashlib.sha256("|".join([prompt, *(r.sha256 for r in refs)]).encode()).hexdigest()[:8], 16) % 2**31

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
