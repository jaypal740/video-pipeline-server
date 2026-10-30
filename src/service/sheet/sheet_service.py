import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor

from src.core.configs import Settings
from src.repository.asset.asset_repository import AssetRepository
from src.repository.run.run_repository import RunRepository
from src.schema.asset.asset_schema import Asset, ReviewState
from src.schema.asset.cost_entry_schema import CostEntry
from src.schema.base_schema import new_id, utcnow
from src.schema.enums import AssetKind, StageName, StageStatus
from src.schema.lesson.lesson_schema import GradeBand
from src.schema.run.stage_state_schema import StageState
from src.schema.storyboard.storyboard_schema import CastKind, CastMember, Place, Storyboard
from src.service.image.fal_image_service import RESOLUTION, FalImageService
from src.service.image.image_check_service import ImageCheckService
from src.service.lesson.lesson_rules import band_rules
from src.service.llm.llm_service import LabelledImage
from src.service.sheet.sheet_prompt import (FACE_VIEWS, HOST_VIEWS, PLATE, Style, aspect_for, plate_prompt,
                                            plate_questions, sheet_prompt, sheet_questions, views_for)
from src.service.sheet.sheet_layout import compose_sheet
from src.service.storage.storage_service import Storage
from src.service.storyboard.storyboard_rules import HOST, host_in

logger = logging.getLogger(__name__)

# Draws per image before it is left for a person: the first, then a new seed, then a new seed with
# more thinking (plain references want minimal thinking: the model then spends less effort on annotations).
MAX_TRIES = 3
URL_SECONDS = 3600
# The outfit the host is approved in; other wardrobe kits are drawn from it on first use.
BASE_KIT = "everyday"
KIT_VIEWS = ["front", "three_quarter", "to_camera"]
EXPRESSION_VIEWS = [view for view in FACE_VIEWS if view != "to_camera"]
USABLE = (ReviewState.PASSED, ReviewState.APPROVED)


class SheetService:
    """Stage 4a: reference sheets. The band's host is drawn once and approved by a person; every other
    sheet in that band is drawn with the host's front view as a style reference, checked against its own
    look, and kept only if it passes. Nothing that fails is ever attached to another picture."""

    def __init__(self, run_repository: RunRepository, asset_repository: AssetRepository, storage: Storage,
                 images: FalImageService, check: ImageCheckService, settings: Settings) -> None:
        self._runs = run_repository
        self._assets = asset_repository
        self._storage = storage
        self._images = images
        self._check = check
        self._settings = settings

    # --- hosts, once per band -------------------------------------------------------------------------

    def host_candidates(self, band: GradeBand, count: int = 3) -> list[Asset]:
        """Front views of the band's host from its description alone, for a person to choose from."""
        host, style = host_in(band, BASE_KIT), Style.of(band_rules(band))
        prompt = sheet_prompt(style, host, "front", reference=None)
        for review in (ReviewState.PENDING, ReviewState.FAILED):
            self._assets.assets.delete_where(run_id=None, target_id=self._target(band, BASE_KIT, "front"), review=review)
        made = []
        questions = sheet_questions(host, "front")

        def draw(n: int):
            reply = self._images.generate(self._settings.host_model, prompt, [], aspect_for(host, "front"), seed=n,
                                          thinking="minimal")
            return reply, self._check.check(reply.data, [], questions)[0].problems()

        with ThreadPoolExecutor(max_workers=count) as pool:
            drawn = list(pool.map(draw, range(1, count + 1)))
        for n, (reply, problems) in enumerate(drawn, start=1):
            asset = self._store(None, f"hosts/{band}/candidates/{n}.png", reply, prompt, [], 1,
                                target_id=self._target(band, BASE_KIT, "front"), view="front",
                                aspect=aspect_for(host, "front"), candidate=n)
            asset.review = ReviewState.FAILED if problems else ReviewState.PENDING
            asset.review_notes = "\n".join(problems) or None
            made.append(self._assets.assets.update(asset))
            logger.info("host %s: candidate %d %s, $%.3f, %.1fs%s", band, n, "failed" if problems else "ok",
                        reply.cost_usd, reply.seconds, "".join(f"\n  - {p}" for p in problems))
        return made

    def approve_host(self, band: GradeBand, asset_id: str) -> list[Asset]:
        """Keep the chosen candidate, then draw the host's other views from it, all in the base kit."""
        chosen = self._assets.assets.get(asset_id)
        if chosen is None or chosen.target_id != self._target(band, BASE_KIT, "front"):
            raise LookupError(f"{asset_id} is not a host candidate for grades {band}")
        if chosen.review == ReviewState.FAILED:
            raise ValueError(f"candidate {asset_id} failed its check ({chosen.review_notes}); pick one that passed")
        for candidate in self._assets.assets.list(target_id=self._target(band, BASE_KIT, "front")):
            candidate.review = ReviewState.APPROVED if candidate.id == asset_id else ReviewState.FAILED
            candidate = self._assets.assets.update(candidate)
            if candidate.id == asset_id:
                chosen = candidate
        host, style = host_in(band, BASE_KIT), Style.of(band_rules(band))
        anchor = [chosen]
        made = [chosen]
        done = self.host_sheets(band)                                  # a second approve draws only what is missing
        for view in HOST_VIEWS[1:]:
            if view in done:
                made.append(done[view])
                continue
            made.append(self._draw(None, f"hosts/{band}/{BASE_KIT}/{view}.png", self._target(band, BASE_KIT, view),
                                   view, sheet_prompt(style, host, view, reference="same"), anchor,
                                   aspect_for(host, view), sheet_questions(host, view), self._settings.host_model))
        self._storage.put(f"hosts/{band}/sheet.png", compose_sheet(
            [(a.params["view"], self._storage.get(a.path)) for a in made]), "image/png")
        return made

    def ensure_kit(self, band: GradeBand, kit: str) -> dict[str, Asset]:
        """The host dressed in one wardrobe kit, drawn once per band from the approved face: the views a
        video needs (front, three-quarter, to camera); the face views are shared with the base kit."""
        found = self.host_sheets(band, kit)
        if all(view in found for view in KIT_VIEWS):
            return found
        base = self.host_sheets(band, BASE_KIT)
        if "front" not in base:
            raise ValueError(f"grades {band} have no approved host yet; run `host draw` and `host approve` first")
        host, style = host_in(band, kit), Style.of(band_rules(band))
        for view in KIT_VIEWS:
            if view not in found:
                found[view] = self._draw(None, f"hosts/{band}/{kit}/{view}.png", self._target(band, kit, view), view,
                                         sheet_prompt(style, host, view, reference="same_person"), [base["front"]],
                                         aspect_for(host, view), sheet_questions(host, view), self._settings.host_model)
        return found

    def recheck_host(self, band: GradeBand) -> list[Asset]:
        """Run the image check again on the host's views waiting for review, with the current questions.
        No image is drawn; a view that now passes becomes usable."""
        host = host_in(band, BASE_KIT)
        changed = []
        for asset in self._assets.assets.list(run_id=None, review=ReviewState.NEEDS_REVIEW):
            if not asset.target_id.startswith(f"host:{band}:"):
                continue
            view = asset.params["view"]
            problems = self._check.check(self._storage.get(asset.path), [], sheet_questions(host, view))[0].problems()
            asset.review = ReviewState.PASSED if not problems else ReviewState.NEEDS_REVIEW
            asset.review_notes = "\n".join(problems) or None
            changed.append(self._assets.assets.update(asset))
            logger.info("host %s/%s rechecked: %s", band, view, "passed" if not problems else "still needs review")
        return changed

    def redraw_host_view(self, band: GradeBand, view: str) -> Asset:
        """Draw one view of the base kit again from the approved front; the old tries are kept as failed."""
        base = self.host_sheets(band)
        if "front" not in base:
            raise ValueError(f"grades {band} have no approved host yet")
        for asset in self._assets.assets.list(run_id=None, target_id=self._target(band, BASE_KIT, view)):
            asset.review = ReviewState.FAILED
            self._assets.assets.update(asset)
        host, style = host_in(band, BASE_KIT), Style.of(band_rules(band))
        made = self._draw(None, f"hosts/{band}/{BASE_KIT}/{view}.png", self._target(band, BASE_KIT, view), view,
                          sheet_prompt(style, host, view, reference="same"), [base["front"]], aspect_for(host, view),
                          sheet_questions(host, view), self._settings.host_model)
        self.compose_host_sheet(band)
        return made

    def compose_host_sheet(self, band: GradeBand) -> None:
        views = self.host_sheets(band)
        tiles = [(v, self._storage.get(views[v].path)) for v in HOST_VIEWS if v in views]
        self._storage.put(f"hosts/{band}/sheet.png", compose_sheet(tiles), "image/png")

    def host_sheet_image(self, band: GradeBand) -> bytes | None:
        """The code-built character sheet of the band's host: every view laid out with labels drawn by us."""
        key = f"hosts/{band}/sheet.png"
        return self._storage.get(key) if self._storage.exists(key) else None

    def host_sheets(self, band: GradeBand, kit: str = BASE_KIT) -> dict[str, Asset]:
        """The band's usable host images by view for one kit: the kit's own views (the to-camera view shows
        the outfit, so it is per kit), plus the base kit's expression views, which every outfit shares.
        Empty until a host is approved."""
        found = {}
        for view in HOST_VIEWS:
            for asset in self._assets.assets.list(target_id=self._target(band, kit, view)):
                if asset.review in USABLE:
                    found[view] = asset
        if kit != BASE_KIT:
            base = self.host_sheets(band, BASE_KIT)
            for view in EXPRESSION_VIEWS:
                if view not in found and view in base:
                    found[view] = base[view]
        return found

    @staticmethod
    def _target(band: GradeBand, kit: str, view: str) -> str:
        return f"host:{band}:{kit}:{view}"

    # --- one run's cast and places ----------------------------------------------------------------------

    def run(self, run_id: str, force: bool = False) -> dict[str, dict[str, Asset]]:
        run = self._runs.runs.get(run_id)
        if run is None:
            raise LookupError(f"Run {run_id} not found")
        board = self._board(run_id)
        band = board.grade_band
        hosts = self.host_sheets(band)
        if any(view not in hosts for view in HOST_VIEWS):
            raise ValueError(f"grades {band} have no approved host yet; run `host draw` and `host approve` first")
        self.ensure_kit(band, board.host_outfit)                      # the host dressed for this video
        style, anchor = Style.of(band_rules(band)), [hosts["front"]]
        jobs = [(m, view) for m in board.cast if m.id != HOST for view in views_for(m)]
        jobs += [(p, PLATE) for p in board.places]
        key = hashlib.sha256((self._settings.sheet_model + "|" + hosts["front"].sha256 + "|" + "|".join(
            f"{t.id}:{t.look}:{v}" for t, v in jobs)).encode()).hexdigest()

        state = self._runs.stage(run_id, StageName.VISUAL_IDENTITY)
        if state is not None and not force and state.status == StageStatus.DONE and state.input_hash == key:
            logger.info("sheets %s: reused, inputs unchanged", run_id[:8])
            return self._sheets_from(state.output)
        state = state or self._runs.stage_states.add(StageState(run_id=run_id, stage=StageName.VISUAL_IDENTITY))
        state.status, state.attempts, state.error = StageStatus.RUNNING, state.attempts + 1, None
        state.input_hash, state.started_at, state.finished_at = key, utcnow(), None
        state = self._runs.stage_states.update(state)
        logger.info("sheets %s: %s, grades %s, %d images to draw", run_id[:8], self._settings.sheet_model, band,
                    len(jobs))
        try:
            results = [self._sheet_job(run_id, style, anchor, *job) for job in jobs]
        except Exception as exc:
            state.status, state.error, state.finished_at = StageStatus.FAILED, str(exc), utcnow()
            self._runs.stage_states.update(state)
            logger.error("sheets %s: failed: %s", run_id[:8], exc)
            raise
        sheets: dict[str, dict[str, Asset]] = {}
        for (target, view), asset in zip(jobs, results):
            sheets.setdefault(target.id, {})[view] = asset
        waiting = [a for a in results if a.review == ReviewState.NEEDS_REVIEW]
        retries = sum(a.params.get("try", 1) - 1 for a in results)
        state.status = StageStatus.WAITING_GATE if waiting else StageStatus.DONE
        state.output = {"sheets": {t: {v: a.id for v, a in views.items()} for t, views in sheets.items()},
                        "needs_review": [a.id for a in waiting]}
        state.finished_at = utcnow()
        self._runs.stage_states.update(state)
        spent = self._assets.run_cost(run_id)
        logger.info("sheets %s: %s, %d images, %d retries, %d need review, $%.2f on the run so far, %.0fs",
                    run_id[:8], state.status, len(results), retries, len(waiting), spent,
                    (state.finished_at - state.started_at).total_seconds())
        return sheets

    def _sheet_job(self, run_id: str, style: Style, anchor: list[Asset], target: CastMember | Place, view: str) -> Asset:
        if view == PLATE:
            return self._draw(run_id, f"runs/{run_id}/sheets/{target.id}/plate.png", target.id, view,
                              plate_prompt(style, target, reference="style"), anchor, aspect_for(None, PLATE),
                              plate_questions(target), self._settings.sheet_model)
        return self._draw(run_id, f"runs/{run_id}/sheets/{target.id}/{view}.png", target.id, view,
                          sheet_prompt(style, target, view, reference="style"), anchor, aspect_for(target, view),
                          sheet_questions(target, view), self._settings.sheet_model)

    # --- drawing, checking, storing ---------------------------------------------------------------------

    def _draw(self, run_id: str | None, key: str, target_id: str, view: str, prompt: str, refs: list[Asset],
              aspect: str, questions: list[str], model: str) -> Asset:
        """Draw, check, and try again on a new seed; after MAX_TRIES the last try is left for review. The seed
        comes from the prompt and references, so the same look draws the same image in any run."""
        base = int(hashlib.sha256("|".join([prompt, *(r.sha256 for r in refs)]).encode()).hexdigest()[:8], 16) % 2**31
        ref_urls = [self._url(r) for r in refs]
        for attempt in range(1, MAX_TRIES + 1):
            seed, thinking = base + attempt - 1, "medium" if attempt == MAX_TRIES else "minimal"
            cache_key = self._cache_key(model, prompt, refs, seed, aspect, thinking)
            cached = self._assets.find_cached(cache_key)
            if cached is not None and cached.review in USABLE:
                logger.info("sheet %s/%s: reused %s", target_id, view, cached.id[:8])
                return self._reuse(run_id, key, cached, target_id, view)
            reply = self._images.generate(model, prompt, ref_urls, aspect, seed=seed, thinking=thinking)
            asset = self._store(run_id, key, reply, prompt, refs, attempt, target_id=target_id, view=view,
                                aspect=aspect, cache_key=cache_key)
            labelled = [LabelledImage(label=f"reference sheet for {r.target_id}", data=self._storage.get(r.path))
                        for r in refs]
            checklist, call = self._check.check(reply.data, labelled, questions)
            self._log_cost(run_id, StageName.VISUAL_IDENTITY, call.provider, call.model, call.cost_usd, target_id,
                           call.input_tokens + call.output_tokens, "tokens")
            problems = checklist.problems()
            if not problems:
                asset.review = ReviewState.PASSED
                self._assets.assets.update(asset)
                logger.info("sheet %s/%s: passed on try %d, $%.3f, %.1fs", target_id, view, attempt, reply.cost_usd,
                            reply.seconds)
                return asset
            asset.review = ReviewState.NEEDS_REVIEW if attempt == MAX_TRIES else ReviewState.FAILED
            asset.review_notes = "\n".join(problems)
            self._assets.assets.update(asset)
            logger.warning("sheet %s/%s: try %d failed %d checks:\n%s", target_id, view, attempt, len(problems),
                           "\n".join(f"  - {p}" for p in problems))
        logger.warning("sheet %s/%s: left for review after %d tries", target_id, view, MAX_TRIES)
        return asset

    def _store(self, run_id: str | None, key: str, reply, prompt: str, refs: list[Asset], attempt: int,
               target_id: str, view: str, aspect: str, cache_key: str | None = None,
               candidate: int | None = None) -> Asset:
        """Store one try under its own file, named by its asset id, so no draw ever overwrites another."""
        asset_id = new_id()
        key = f"{key.removesuffix('.png')}/{attempt}-{asset_id[:8]}.png"
        self._storage.put(key, reply.data, reply.content_type)
        params = {"view": view, "try": attempt, "aspect": aspect, "resolution": RESOLUTION,
                  "refs": [r.id for r in refs], "seconds": reply.seconds}
        if candidate is not None:
            params["candidate"] = candidate
        asset = self._assets.assets.add(Asset(
            id=asset_id, run_id=run_id, stage=StageName.VISUAL_IDENTITY, kind=AssetKind.IMAGE, path=key,
            sha256=hashlib.sha256(reply.data).hexdigest(), target_id=target_id, cache_key=cache_key,
            provider=self._images.name, model=reply.model, seed=reply.seed, prompt=prompt, params=params,
            cost_usd=reply.cost_usd))
        self._log_cost(run_id, StageName.VISUAL_IDENTITY, self._images.name, reply.model, reply.cost_usd, target_id,
                       1, "image")
        return asset

    def _reuse(self, run_id: str | None, key: str, cached: Asset, target_id: str, view: str) -> Asset:
        """A row for this run pointing at the cached file; nothing is copied or overwritten."""
        return self._assets.assets.add(Asset(
            run_id=run_id, stage=StageName.VISUAL_IDENTITY, kind=AssetKind.IMAGE, path=cached.path,
            sha256=cached.sha256,
            target_id=target_id, cache_key=cached.cache_key, provider=cached.provider, model=cached.model,
            seed=cached.seed, prompt=cached.prompt, params={**cached.params, "view": view, "reused_from": cached.id},
            cost_usd=0.0, review=ReviewState.PASSED))

    def _url(self, asset: Asset) -> str:
        return self._storage.url(asset.path, URL_SECONDS) or self._images.upload(self._storage.get(asset.path),
                                                                                 "image/png")

    @staticmethod
    def _cache_key(model: str, prompt: str, refs: list[Asset], seed: int, aspect: str, thinking: str) -> str:
        parts = [model, prompt, *(r.sha256 for r in refs), str(seed), aspect, RESOLUTION, thinking]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    def _log_cost(self, run_id: str | None, stage: StageName, provider: str, model: str, cost: float,
                  target_id: str, units: float, unit: str) -> None:
        if run_id is not None:                                  # host images carry their cost on the asset
            self._assets.cost_entries.add(CostEntry(run_id=run_id, stage=stage, provider=provider, model=model,
                                                    cost_usd=cost, target_id=target_id, units=units, unit=unit))

    def _board(self, run_id: str) -> Storyboard:
        state = self._runs.stage(run_id, StageName.STORYBOARD)
        if state is None or state.status != StageStatus.DONE:
            raise ValueError(f"Run {run_id} has no storyboard; run the storyboard stage first")
        return Storyboard(**state.output)

    def _sheets_from(self, output: dict) -> dict[str, dict[str, Asset]]:
        return {t: {v: self._assets.assets.get(a) for v, a in views.items()} for t, views in output["sheets"].items()}
