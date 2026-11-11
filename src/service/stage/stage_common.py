import logging
import threading

from src.repository.asset.asset_repository import AssetRepository
from src.repository.run.run_repository import RunRepository
from src.schema.asset.asset_schema import Asset, ReviewState
from src.schema.asset.cost_entry_schema import CostEntry
from src.schema.base_schema import utcnow
from src.schema.enums import StageName, StageStatus
from src.schema.image.image_check_schema import Brief, Correction
from src.schema.run.stage_state_schema import StageState
from src.schema.storyboard.script_schema import AnimationScript
from src.schema.storyboard.storyboard_schema import Storyboard
from src.service.keyframe.keyframe_prompt import CORRECTION, check_correction, correction_request
from src.service.llm.llm_service import LlmCall, LlmError, LlmService
from src.service.storage.storage_service import Storage

# Steps the stages share. Each takes the calling stage's logger, so log lines name the stage as before.

URL_SECONDS = 3600
PASSED = (ReviewState.PASSED, ReviewState.APPROVED)          # a try that met the bar; USABLE adds used_best
# The stop-loss looks at the first this-many finished shots.
STOP_LOSS_AFTER = 8


def storyboard_of(runs: RunRepository, run_id: str) -> Storyboard:
    state = runs.stage(run_id, StageName.STORYBOARD)
    if state is None or state.status != StageStatus.DONE:
        raise ValueError(f"Run {run_id} has no storyboard; run the storyboard stage first")
    return Storyboard(**state.output)


def script_of(runs: RunRepository, run_id: str) -> AnimationScript:
    state = runs.stage(run_id, StageName.STORYBOARD)
    if state is None or state.status != StageStatus.DONE:
        raise ValueError(f"Run {run_id} has no animation script; run the storyboard stage first")
    return AnimationScript(**state.output)


def asset_url(storage: Storage, upload, asset: Asset) -> str:
    """A link the model provider can fetch: the store's own link, else the picture uploaded to the provider."""
    return storage.url(asset.path, URL_SECONDS) or upload(storage.get(asset.path), "image/png")


def add_cost(assets: AssetRepository, run_id: str | None, stage: StageName, provider: str, model: str, cost: float,
             target_id: str, units: float, unit: str) -> None:
    assets.cost_entries.add(CostEntry(run_id=run_id, stage=stage, provider=provider, model=model, cost_usd=cost,
                                      target_id=target_id, units=units, unit=unit))


def add_llm_cost(assets: AssetRepository, run_id: str, stage: StageName, call: LlmCall) -> float:
    if call.attempts:
        assets.cost_entries.add(CostEntry(
            run_id=run_id, stage=stage, provider=call.provider, model=call.model,
            cost_usd=call.cost_usd, units=call.input_tokens + call.output_tokens, unit="tokens"))
    return call.cost_usd


def fail_stage(runs: RunRepository, state: StageState, exc: Exception, log: logging.Logger, what: str) -> None:
    state.status, state.error, state.finished_at = StageStatus.FAILED, str(exc) or type(exc).__name__, utcnow()
    runs.stage_states.update(state)
    log.error("%s %s: failed: %s", what, state.run_id[:8], state.error)


def keep_best(assets: AssetRepository, floor: float, log: logging.Logger, what: str, shot_id: str,
              tries: list[Asset], errors: dict[str, str], pool: list[Asset], pool_only: bool) -> Asset | None:
    """Which of a shot's tries the stage uses: the best passed one; else the best-scoring try of `pool` (all
    tries when the pool is empty) at or above the floor, marked used_best; else none, and the shot is recorded
    as failed. With pool_only, a try from outside an empty pool is never used. Rows are updated to match."""
    passed = [a for a in tries if a.review in PASSED]
    if passed:
        return max(passed, key=lambda a: a.params.get("score") or 0)
    best = max(pool or tries, key=lambda a: a.params.get("score") or 0)
    for a in tries:
        review = (ReviewState.USED_BEST if a is best and (pool or not pool_only)
                  and (a.params.get("score") or 0) >= floor else ReviewState.FAILED)
        if a.review != review:
            a.review = a.review if a.review == ReviewState.APPROVED else review
            assets.assets.update(a)
    if best.review == ReviewState.USED_BEST:
        log.warning("%s %s: no try passed; using the best, %s/10: %s", what, shot_id, best.params.get("score"),
                    (best.review_notes or "").splitlines()[0] if best.review_notes else "")
        return best
    errors[shot_id] = (f"failed every draw; best {best.params.get('score')}/10: "
                       + "; ".join((best.review_notes or "").splitlines()))
    log.error("%s %s: %s", what, shot_id, errors[shot_id])
    return None


def write_correction(llm: LlmService, model: str, log: logging.Logger, what: str, shot_id: str, prompt: str,
                     brief: Brief, findings: list[str]) -> tuple[str | None, LlmCall | None]:
    """A short correction for the retry, from the failed prompt, the opening moment and the judge's
    findings; when none can be written the retry is drawn from the plain prompt."""
    try:
        fix, call = llm.parse(model, CORRECTION, correction_request(prompt, brief.moment or "", findings), Correction,
                              check=check_correction, effort="low")
    except LlmError as exc:
        log.warning("%s %s: no correction, retrying the plain prompt: %s", what, shot_id, exc)
        return None, exc.call
    log.info("%s %s: retry corrected: %s", what, shot_id, fix.text)
    return fix.text, call


class StopLoss:
    """After the first STOP_LOSS_AFTER finished shots, too many failures pause the stage: `reason` is set
    and the stage draws nothing more. Safe across the drawing threads."""

    def __init__(self, share: float, what: str) -> None:
        self._lock, self._finished, self._failed, self._share, self._what = threading.Lock(), 0, 0, share, what
        self.reason: str | None = None

    def count(self, passed: bool) -> None:
        with self._lock:
            self._finished += 1
            self._failed += not passed
            if self._finished == STOP_LOSS_AFTER and self._failed > STOP_LOSS_AFTER * self._share \
                    and not self.reason:
                self.reason = (f"stopped: {self._failed} of the first {self._finished} {self._what} failed every "
                               "draw; fix the cause before drawing more")
