import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import NamedTuple

from pydantic import BaseModel

from src.core.configs import Settings
from src.repository.asset.asset_repository import AssetRepository
from src.repository.run.run_repository import RunRepository
from src.schema.asset.asset_schema import Asset, ReviewState
from src.schema.base_schema import new_id, utcnow
from src.schema.enums import AssetKind, StageName, StageStatus
from src.schema.image.image_check_schema import Brief
from src.schema.run.stage_state_schema import StageState
from src.schema.storyboard.storyboard_schema import (CameraMove, CastMember, Continuity, Moves, Place, Shot,
                                                     ShotKind, Storyboard)
from src.service.image.fal_image_service import RESOLUTION, FalImageService, ImageReply
from src.service.image.image_check_service import ImageCheckService
from src.service.lesson.lesson_rules import band_rules
from src.service.llm.llm_service import LabelledImage, LlmCall, LlmService
from src.service.keyframe.keyframe_prompt import (LINK_LABEL, START_LABEL, end_brief, end_prompt, keyframe_brief,
                                                  keyframe_prompt, linked_brief, linked_prompt, ref_label, view_for)
from src.service.sheet.sheet_prompt import PLATE, Style
from src.service.image.image_prices import image_model
from src.service.sheet.sheet_service import USABLE, SheetService
from src.service.stage.stage_common import (PASSED, StopLoss, add_cost, asset_url, keep_best, storyboard_of,
                                            write_correction)
from src.service.storage.storage_service import Storage
from src.service.storyboard.storyboard_rules import HOST

logger = logging.getLogger(__name__)

ASPECT = "16:9"


# What a keyframe row is: a shot's opening drawn from the sheets, an opening drawn from the picture the shot
# before ended on, or a shot's ending drawn from its own opening.
START, LINKED, END = "keyframe", "linked_keyframe", "end_keyframe"


class Plan(NamedTuple):
    """One picture to draw. key is the shot id for an opening and "<shot id>:end" for an ending; labels name
    each reference for the judge, in the order of refs."""

    key: str
    shot: Shot
    view: str
    prompt: str
    refs: list[Asset]
    labels: list[str]
    brief: Brief


def chains_of(shots: list[Shot]) -> list[list[Shot]]:
    """Runs of shots that carry on from each other: the same place, shot after shot. A new scene, a sung
    shot (its own picture, for lip-sync) or a chorus repeat starts a new chain; a repeat is in none."""
    chains: list[list[Shot]] = []
    prev: Shot | None = None
    for shot in shots:
        if shot.repeat_of is not None:
            prev = None
            continue
        joins = (prev is not None and shot.place == prev.place and shot.continuity != Continuity.NEW_SCENE
                 and ShotKind.SUNG not in (shot.kind, prev.kind))
        if joins:
            chains[-1].append(shot)
        else:
            chains.append([shot])
        prev = shot
    return chains


def needs_end(shot: Shot) -> bool:
    """A shot gets an ending picture when something in it changes; a sung shot is animated by lip-sync and a
    camera shot by code, from the opening alone."""
    return shot.kind != ShotKind.SUNG and shot.moves == Moves.SUBJECT


def same_framing(prev: Shot, shot: Shot) -> bool:
    """The next shot opens on exactly the picture the one before ended on: a still camera, the same framing."""
    return (prev.camera.move == CameraMove.STATIC and (prev.size, prev.angle, prev.position)
            == (shot.size, shot.angle, shot.position))


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
    notes: list[str]
    cache_key: str
    reply: ImageReply
    problems: list[str]
    check: LlmCall
    second: LlmCall | None = None               # the stronger judge's call, when every try failed


class KeyframeService:
    """Stage 4b: the pictures the clips are animated between. Shots in the same place form a chain: the
    first opening is drawn from start_state with the place plate and the subjects' sheets attached; each
    later opening is drawn from the picture the shot before ended on, so the cut joins up; a shot that
    changes also gets its ending, drawn from its opening. Workers draw and check over the network; only the main thread
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
        board = storyboard_of(self._runs, run_id)
        sheets = self._usable_sheets(run_id, board)
        style, model = Style.of(band_rules(board.grade_band)), self._settings.keyframe_model
        cast = {c.id: c for c in board.cast}
        places = {p.id: p for p in board.places}
        chains = chains_of(board.shots)
        parts = [model, ";".join(",".join(s.id for s in chain) for chain in chains)]
        for shot in (s for chain in chains for s in chain):
            subjects, refs = [cast[s] for s in shot.subjects], self._refs_for(shot, cast, sheets)
            parts.append(f"{shot.id}:{keyframe_prompt(style, shot, places[shot.place], subjects)}:"
                         f"{linked_prompt(style, shot, subjects)}:{end_prompt(style, shot) if needs_end(shot) else ''}:"
                         f"{','.join(r.sha256 for r in refs) if refs else 'blocked'}")
        key = hashlib.sha256("|".join(parts).encode()).hexdigest()

        state = self._runs.stage(run_id, StageName.KEYFRAMES)
        if state is not None and not force and state.status == StageStatus.DONE and state.input_hash == key:
            logger.info("keyframes %s: reused, inputs unchanged", run_id[:8])
            return {s: self._assets.assets.get(a) for s, a in state.output["keyframes"].items()}
        state = state or self._runs.stage_states.add(StageState(run_id=run_id, stage=StageName.KEYFRAMES))
        state.status, state.attempts, state.error = StageStatus.RUNNING, state.attempts + 1, None
        state.input_hash, state.started_at, state.finished_at = key, utcnow(), None
        state = self._runs.stage_states.update(state)
        shots = sum(len(c) for c in chains)
        logger.info("keyframes %s: %s, %d shots in %d chains, %d repeats reused", run_id[:8], model, shots,
                    len(chains), len(board.shots) - shots)
        self._errors: dict[str, str] = {}
        self._stop = StopLoss(self._settings.image_stop_loss, "shots")
        try:
            made, ends, blocked = self._draw_chains(run_id, model, style, chains, cast, places, sheets)
        except Exception as exc:
            state.status, state.error, state.finished_at = StageStatus.FAILED, str(exc), utcnow()
            self._runs.stage_states.update(state)
            logger.error("keyframes %s: failed: %s", run_id[:8], exc)
            raise
        for shot in board.shots:                                    # a chorus repeat shows its original's pictures
            if shot.repeat_of in made:
                made[shot.id] = made[shot.repeat_of]
            if shot.repeat_of in ends:
                ends[shot.id] = ends[shot.repeat_of]
        self._finish(run_id, state, made, ends, blocked, shots)
        return made

    def _finish(self, run_id: str, state: StageState, made: dict[str, Asset], ends: dict[str, Asset],
                blocked: list[str], drawn: int) -> None:
        """No person decides here: a shot with a usable opening is done, any other shot fails the stage. A
        missing ending only breaks the chain there; its clip is animated from the opening alone."""
        used_best = sorted({a.id for a in [*made.values(), *ends.values()] if a.review == ReviewState.USED_BEST})
        no_end = sorted(k.removesuffix(":end") for k in self._errors if k.endswith(":end"))
        blocked = sorted(set(blocked) | {k for k in self._errors if not k.endswith(":end")})
        state.status = StageStatus.FAILED if blocked else StageStatus.DONE
        state.error = ("; ".join(f"{s}: {self._errors.get(s, 'no usable reference')}" for s in blocked) or None)
        state.output = {"keyframes": {s: a.id for s, a in made.items()},
                        "end_keyframes": {s: a.id for s, a in ends.items()}, "used_best": used_best,
                        "blocked": blocked, "no_end": no_end, "errors": self._errors}
        state.finished_at = utcnow()
        self._runs.stage_states.update(state)
        retries = sum(a.params.get("try", 1) - 1 for a in {a.id: a for a in [*made.values(), *ends.values()]}.values())
        logger.log(logging.ERROR if blocked else logging.INFO,
                   "keyframes %s: %s, %d shots, %d endings, %d retries, %d used best, %d failed, %d without an ending, "
                   "$%.2f on the run so far, %.0fs", run_id[:8], state.status, drawn, len(ends), retries,
                   len(used_best), len(blocked), len(no_end), self._assets.run_cost(run_id),
                   (state.finished_at - state.started_at).total_seconds())

    def recheck(self, run_id: str) -> dict[str, Asset]:
        """Judge every keyframe of this run that did not pass again, with the current brief; no image is drawn.
        The stage then takes the best passed try of each shot, or waits for a person as before."""
        board = storyboard_of(self._runs, run_id)
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
                 for t in self._assets.assets.list(run_id=drawn_in, stage=StageName.KEYFRAMES, target_id=shot_id)
                 if t.params.get("view", START) == START]           # linked openings and endings are not rechecked
        self._errors = {k: v for k, v in state.output.get("errors", {}).items() if k not in shots
                        or shots[k].repeat_of is not None}
        inputs: dict[str, tuple[Brief, list[LabelledImage]]] = {}
        for asset in tries:                                     # every try of each shot, in the run that drew it
            if asset.review in PASSED:
                continue
            shot = shots[asset.target_id]
            brief = keyframe_brief(shot, places[shot.place], [cast[s] for s in shot.subjects])
            refs = [self._assets.assets.get(r) for r in asset.params["refs"]]
            labelled = [LabelledImage(label=ref_label(kind, name), data=self._storage.get(r.path))
                        for (kind, name), r in zip([('place', shot.place), *(('cast member', s) for s in shot.subjects)], refs)]
            inputs[asset.id] = brief, labelled
            self._recheck_one(run_id, asset, brief, labelled, self._settings.image_check_model)
        made = {s: a for s, i in state.output["keyframes"].items() if (a := self._assets.assets.get(i)) is not None}
        ends = {s: a for s, i in state.output.get("end_keyframes", {}).items()
                if (a := self._assets.assets.get(i)) is not None}
        by_shot: dict[str, list[Asset]] = {}
        for asset in tries:
            by_shot.setdefault(asset.target_id, []).append(asset)
        second = self._settings.image_second_opinion_model
        for shot_tries in by_shot.values():                     # as on a fresh run: no try passed, the best one
            judged = [a for a in shot_tries if a.id in inputs]  # gets the second opinion, and its verdict stands
            if second and judged and not any(a.review in PASSED for a in shot_tries):
                best = max(judged, key=lambda a: a.params.get("score") or 0)
                self._recheck_one(run_id, best, *inputs[best.id], second)
        for shot_id, shot_tries in by_shot.items():
            kept = self._keep(shot_id, shot_tries)
            if kept is not None:
                made[shot_id] = kept
        for shot in board.shots:
            if shot.repeat_of in made:
                made[shot.id] = made[shot.repeat_of]
        self._errors.update({k: v for k, v in state.output.get("errors", {}).items() if k.endswith(":end")})
        blocked = [s for s in state.output.get("blocked", []) if s not in by_shot]
        state.started_at = state.started_at or utcnow()
        self._finish(run_id, state, made, ends, blocked, len(by_shot))
        return made

    def _recheck_one(self, run_id: str, asset: Asset, brief: Brief, labelled: list[LabelledImage], model: str) -> None:
        """Judge one stored try again with `model` and write the verdict, and who gave it, onto its row."""
        checklist, call = self._check.check(self._storage.get(asset.path), labelled, brief, model=model)
        problems = self._check.problems(checklist, brief)
        add_cost(self._assets, run_id, StageName.KEYFRAMES, call.provider, call.model, call.cost_usd,
                 asset.target_id, call.input_tokens + call.output_tokens, "tokens")
        asset.params = {**asset.params, "score": checklist.score, "judged_by": model,
                        "asked_for": [a.model_dump() for a in checklist.asked_for]}
        asset.review = ReviewState.PASSED if not problems else ReviewState.FAILED
        asset.review_notes = "\n".join(self._check.notes(checklist)) or None
        self._assets.assets.update(asset)
        logger.info("keyframe %s try %d rechecked by %s: %s, %d/10", asset.target_id, asset.params["try"], model,
                    asset.review, checklist.score)

    def _keep(self, shot_id: str, tries: list[Asset]) -> Asset | None:
        """Which of a shot's tries the stage uses (see keep_best). Once the second-opinion judge has judged a
        try, only its verdict counts for the floor."""
        second = self._settings.image_second_opinion_model
        judged = [a for a in tries if second and a.params.get("judged_by") == second]
        return keep_best(self._assets, self._settings.image_floor_score, logger, "keyframe", shot_id, tries,
                         self._errors, judged, pool_only=False)

    # --- drawing ------------------------------------------------------------------------------------------

    def _draw_chains(self, run_id: str, model: str, style: Style, chains: list[list[Shot]], cast: dict[str, CastMember],
                     places: dict[str, Place], sheets: dict[str, dict[str, Asset]]
                     ) -> tuple[dict[str, Asset], dict[str, Asset], list[str]]:
        """Every chain moves forward one picture per round, and each round is drawn in parallel; a picture is
        planned only once the one it is drawn from is stored. Returns the openings and endings by shot id,
        and the shots blocked by their sheets."""
        made: dict[str, Asset] = {}
        ends: dict[str, Asset] = {}
        blocked: list[str] = []
        progress = [[chain, 0, START] for chain in chains]          # where each chain is: shot index, picture
        while True:
            round_ = [(plan, p) for p in progress
                      if (plan := self._next_plan(p, style, cast, places, sheets, made, ends, blocked)) is not None]
            if not round_:
                return made, ends, blocked
            drawn = self._draw_all(run_id, model, [plan for plan, _ in round_])
            for plan, p in round_:
                if plan.view == END:
                    if plan.key in drawn:
                        ends[plan.shot.id] = drawn[plan.key]
                    p[1], p[2] = p[1] + 1, START
                elif plan.key in drawn:
                    made[plan.shot.id] = drawn[plan.key]
                    p[2] = END
                else:                                               # failed: the next shot starts a new chain
                    p[1], p[2] = p[1] + 1, START

    def _next_plan(self, p: list, style: Style, cast: dict[str, CastMember], places: dict[str, Place],
                   sheets: dict[str, dict[str, Asset]], made: dict[str, Asset], ends: dict[str, Asset],
                   blocked: list[str]) -> Plan | None:
        """The next picture a chain needs drawn, or None when it is done. Steps that draw nothing (a shot that
        opens on the previous ending, a shot with no ending) are taken on the way."""
        chain, i, view = p
        while i < len(chain):
            shot = chain[i]
            subjects = [cast[s] for s in shot.subjects]
            if view == END:
                if shot.id in made and needs_end(shot):
                    p[1], p[2] = i, END
                    return Plan(f"{shot.id}:end", shot, END, end_prompt(style, shot), [made[shot.id]], [START_LABEL],
                                end_brief(shot))
                i, view = i + 1, START
                continue
            prev = chain[i - 1] if i else None
            link = (ends.get(prev.id) or made.get(prev.id)) if prev is not None else None
            if link is not None and prev.id in ends and same_framing(prev, shot):
                made[shot.id] = link                                # the cut opens on the very same picture
                logger.info("keyframe %s: opens on %s's ending", shot.id, prev.id)
                view = END
                continue
            refs = self._refs_for(shot, cast, sheets)
            if refs is None:
                blocked.append(shot.id)
                i += 1
                continue
            labels = [ref_label("cast member", s) for s in shot.subjects]
            p[1], p[2] = i, START
            if link is None:
                return Plan(shot.id, shot, START, keyframe_prompt(style, shot, places[shot.place], subjects), refs,
                            [ref_label("place", shot.place), *labels], keyframe_brief(shot, places[shot.place], subjects))
            return Plan(shot.id, shot, LINKED, linked_prompt(style, shot, subjects), [link, *refs[1:]],
                        [LINK_LABEL, *labels], linked_brief(shot, places[shot.place], subjects))
        p[1] = i
        return None

    def _draw_all(self, run_id: str, model: str, plans: list[Plan]) -> dict[str, Asset]:
        made: dict[str, Asset] = {}
        todo = []
        for plan in plans:
            seeds = self._tries(model, plan.prompt, plan.refs, self._base_seed(plan.prompt, plan.refs))
            cached = {key: c for *_, key in seeds if (c := self._assets.find_cached(key)) is not None}
            tried = list(cached.values())                       # a seed is never drawn twice: same seed, same picture
            usable = next((c for c in tried if c.review in USABLE), None)
            left = [s for s in seeds if s[3] not in cached]
            if usable is not None:                              # the run that drew it keeps it; no new row
                logger.info("keyframe %s: reused %s (%s)", plan.key, usable.id[:8], usable.review)
                made[plan.key] = usable
            elif not left:                                      # every seed tried, none good enough
                self._keep(plan.key, tried)
            else:                                               # the last try drawn so far says what to correct
                last = max(tried, key=lambda c: c.params.get("try", 1), default=None)
                findings = (last.review_notes or "").splitlines() if last is not None else []
                todo.append((plan, left, findings))
        if not todo:
            return made
        self._guard_spend(run_id, model, len(todo))
        ref_urls = {r.id: asset_url(self._storage, self._images.upload, r) for plan, *_ in todo for r in plan.refs}
        ref_bytes = {r.id: self._storage.get(r.path) for plan, *_ in todo for r in plan.refs}
        with ThreadPoolExecutor(max_workers=self._settings.image_workers) as pool:
            futures = {pool.submit(self._attempts, model, ref_urls, ref_bytes, *job): job[0] for job in todo}
            for future in as_completed(futures):               # stored the moment each picture finishes
                plan = futures[future]
                try:
                    tries = future.result()
                except Exception as exc:                        # one picture's error never loses the others
                    logger.error("keyframe %s: error, left for review: %s", plan.key, exc)
                    self._errors[plan.key] = str(exc)
                    continue
                if not tries:                                   # skipped: the stop-loss had fired
                    self._errors[plan.key] = self._stop.reason
                    continue
                kept = self._store_attempts(run_id, plan, tries)
                if kept is not None:
                    made[plan.key] = kept
        if self._stop.reason:
            logger.error("keyframes %s: %s; %d pictures not drawn", run_id[:8], self._stop.reason,
                         sum(1 for e in self._errors.values() if e == self._stop.reason))
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

    def _attempts(self, model: str, ref_urls: dict, ref_bytes: dict, plan: Plan,
                  seeds: list[tuple[int, int, str, str]], findings: list[str]) -> list[Attempt]:
        """Network only: draw, check, retry on a new seed with a correction written from the last try's
        findings. No database access in here. The workers keep the stop-loss count themselves, so no draw
        starts once it has fired, whatever the timing."""
        if self._stop.reason:
            return []
        shot, prompt, brief = plan.shot, plan.prompt, plan.brief
        urls = [ref_urls[r.id] for r in plan.refs]
        labelled = [LabelledImage(label=label, data=ref_bytes[r.id]) for label, r in zip(plan.labels, plan.refs)]
        tries = []
        for attempt, seed, thinking, cache_key in seeds:
            correction, fix = (write_correction(self._llm, self._settings.image_check_model, logger, "keyframe",
                                                shot.id, prompt, brief, findings) if findings else (None, None))
            drawn = f"{prompt} {correction}" if correction else prompt
            reply = self._images.generate(model, drawn, urls, ASPECT, seed=seed, thinking=thinking)
            checklist, call = self._check.check(reply.data, labelled, brief)
            problems, findings = self._check.problems(checklist, brief), self._check.notes(checklist)
            tries.append(Attempt(attempt=attempt, seed=seed, thinking=thinking, prompt=drawn, correction=correction,
                                 fix=fix, score=checklist.score,
                                 asked_for=[a.model_dump() for a in checklist.asked_for], notes=findings,
                                 cache_key=cache_key, reply=reply,
                                 problems=problems, check=call))
            if not problems:
                logger.info("keyframe %s: passed on try %d, $%.3f, %.1fs", plan.key, attempt, reply.cost_usd,
                            reply.seconds)
                self._stop.count(passed=True)
                return tries
            logger.warning("keyframe %s: try %d failed %d checks:\n%s", plan.key, attempt, len(problems),
                           "\n".join(f"  - {p}" for p in problems))
        if self._second_opinion(plan, tries, labelled):
            self._stop.count(passed=True)
            return tries
        logger.warning("keyframe %s: left for review after %d tries", plan.key, len(seeds))
        self._stop.count(passed=False)
        return tries

    def _second_opinion(self, plan: Plan, tries: list[Attempt], labelled: list[LabelledImage]) -> bool:
        """Every try failed: the best one is judged once more by a stronger model, which catches a judge that
        holds a picture to everyday looks the moment asks against. When it passes, its verdict replaces the
        first judge's on that try."""
        model = self._settings.image_second_opinion_model
        if not model or not tries:
            return False
        best = max(tries, key=lambda t: t.score)
        checklist, call = self._check.check(best.reply.data, labelled, plan.brief, model=model)
        problems = self._check.problems(checklist, plan.brief)
        best.second, best.problems, best.score = call, problems, checklist.score      # its verdict stands, either way
        best.notes, best.asked_for = self._check.notes(checklist), [a.model_dump() for a in checklist.asked_for]
        if problems:
            logger.warning("keyframe %s: second opinion from %s agrees, try %d fails:\n%s", plan.key, model,
                           best.attempt, "\n".join(f"  - {p}" for p in problems))
            return False
        logger.info("keyframe %s: try %d passed on a second opinion from %s, %s/10", plan.key, best.attempt, model,
                    checklist.score)
        return True

    def _store_attempts(self, run_id: str, plan: Plan, tries: list[Attempt]) -> Asset | None:
        shot, refs = plan.shot, plan.refs
        prefix = "" if plan.view == START else f"{plan.view.removesuffix('_keyframe')}-"
        stored = []
        for t in tries:
            asset_id = new_id()
            path = f"runs/{run_id}/keyframes/{shot.id}/{prefix}{t.attempt}-{asset_id[:8]}.png"
            self._storage.put(path, t.reply.data, t.reply.content_type)
            review = ReviewState.PASSED if not t.problems else ReviewState.FAILED
            asset = self._assets.assets.add(Asset(
                id=asset_id, run_id=run_id, stage=StageName.KEYFRAMES, kind=AssetKind.IMAGE, path=path,
                sha256=hashlib.sha256(t.reply.data).hexdigest(), target_id=shot.id, cache_key=t.cache_key,
                provider=self._images.name, model=t.reply.model, seed=t.reply.seed, prompt=t.prompt,
                params={"view": plan.view, "try": t.attempt, "aspect": ASPECT, "resolution": RESOLUTION,
                        "thinking": t.thinking, "refs": [r.id for r in refs], "labels": plan.labels,
                        "seconds": t.reply.seconds, "judged_by": (t.second or t.check).model,
                        "score": t.score, "asked_for": t.asked_for, "correction": t.correction},
                cost_usd=t.reply.cost_usd, review=review, review_notes="\n".join(t.notes) or None))
            add_cost(self._assets, run_id, StageName.KEYFRAMES, self._images.name, t.reply.model, t.reply.cost_usd,
                     shot.id, 1, "image")
            for call in (c for c in (t.check, t.fix, t.second) if c is not None):
                add_cost(self._assets, run_id, StageName.KEYFRAMES, call.provider, call.model, call.cost_usd, shot.id,
                         call.input_tokens + call.output_tokens, "tokens")
            stored.append(asset)
        return self._keep(plan.key, stored)

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
