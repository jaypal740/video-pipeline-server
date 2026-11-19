import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

from src.repository.asset.asset_repository import AssetRepository
from src.repository.run.run_repository import RunRepository
from src.repository.song.song_repository import SongRepository
from src.schema.asset.asset_schema import Asset, ReviewState
from src.schema.asset.cost_entry_schema import CostEntry
from src.schema.base_schema import utcnow
from src.schema.enums import FilmType, StageName, StageStatus
from src.schema.image.image_check_schema import Brief, Correction
from src.schema.run.run_schema import Run
from src.schema.run.stage_state_schema import StageState
from src.schema.song.song_analysis_schema import SongAnalysis
from src.schema.storyboard.storyboard_schema import Storyboard
from src.service.intake.intake_jamwise import export_beside
from src.service.keyframe.keyframe_prompt import CORRECTION, check_correction, correction_request
from src.service.llm.llm_service import LlmCall, LlmError, LlmService
from src.service.storage.storage_service import Storage

T = TypeVar("T")

# Steps the stages share. Each takes the calling stage's logger, so log lines name the stage as before.

URL_SECONDS = 3600
PASSED = (ReviewState.PASSED, ReviewState.APPROVED)          # a try that met the bar; USABLE adds used_best
# The stop-loss looks at the first this-many finished shots.
STOP_LOSS_AFTER = 8
# The film types the realistic stages make; 2d films have their own stages under two_d.
REALISTIC = frozenset({FilmType.NARRATION, FilmType.MUSIC_VIDEO})


def film_type_of(runs: RunRepository, run_id: str) -> FilmType | None:
    """The run's film type, from its lesson: a 2d lesson has beats, a realistic one says its type. None when
    there is no lesson yet, or a realistic lesson stored before film types existed."""
    state = runs.stage(run_id, StageName.LESSON)
    output = state.output if state is not None else {}
    if "beats" in output:
        return FilmType.TWO_D
    return FilmType(output["film_type"]) if output.get("film_type") else None


def require_film_type(runs: RunRepository, run_id: str, allowed: frozenset[FilmType]) -> None:
    """Stops a stage before it spends anything on a run of a film type it does not make. A run with no lesson
    yet is accepted by any stage; its lesson then fixes its type."""
    state = runs.stage(run_id, StageName.LESSON)
    if state is None or not state.output:
        return
    found = film_type_of(runs, run_id)
    if found in allowed or (found is None and allowed == REALISTIC):
        return
    hint = "run 2d <stage>" if found == FilmType.TWO_D else "run <stage>"
    raise ValueError(f"run {run_id[:8]} is a {found or 'realistic'} film; use `{hint}`")


def run_of(runs: RunRepository, run_id: str) -> Run:
    run = runs.runs.get(run_id)
    if run is None:
        raise LookupError(f"Run {run_id} not found")
    return run


def analysis_of(songs: SongRepository, run: Run) -> SongAnalysis:
    song = songs.songs.get(run.song_id)
    analysis = songs.analyses.get(song.analysis_id) if song is not None and song.analysis_id is not None else None
    if analysis is None:
        raise LookupError(f"Run {run.id}'s song has not been analysed; run intake on it first")
    return analysis


def export_of(songs: SongRepository, run: Run) -> dict | None:
    """The song's JamWise export, if it came from JamWise."""
    return export_beside(songs.songs.get(run.song_id).audio_path)


def start_stage(runs: RunRepository, state: StageState | None, run_id: str, stage: StageName, key: str) -> StageState:
    """Marks the stage running for the inputs `key`: a first run adds its row, a re-run counts another attempt."""
    state = state or runs.stage_states.add(StageState(run_id=run_id, stage=stage))
    state.status, state.attempts, state.error = StageStatus.RUNNING, state.attempts + 1, None
    state.input_hash, state.started_at, state.finished_at = key, utcnow(), None
    return runs.stage_states.update(state)


def storyboard_of(runs: RunRepository, run_id: str) -> Storyboard:
    require_film_type(runs, run_id, REALISTIC)
    state = runs.stage(run_id, StageName.STORYBOARD)
    if state is None or state.status != StageStatus.DONE:
        raise ValueError(f"Run {run_id} has no storyboard; run the storyboard stage first")
    return Storyboard(**state.output)


def asset_url(storage: Storage, upload, asset: Asset) -> str:
    """A link the model provider can fetch: the store's own link, else the picture uploaded to the provider."""
    return storage.url(asset.path, URL_SECONDS) or upload(storage.get(asset.path), "image/png")


def asset_urls(storage: Storage, upload, assets: list[Asset], workers: int) -> dict[str, str]:
    """A link for each picture, by asset id, uploaded several at once: one at a time, a film's pictures took
    longer to upload than its clips took to make."""
    unique = list({a.id: a for a in assets}.values())
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return dict(zip((a.id for a in unique), pool.map(lambda a: asset_url(storage, upload, a), unique)))


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


def parse_stage(llm: LlmService, assets: AssetRepository, runs: RunRepository, state: StageState, run_id: str,
                stage: StageName, what: str, log: logging.Logger, model: str, system: str, prompt: str,
                output_type: type[T], check: Callable[[T], list[str]]) -> tuple[T, LlmCall]:
    """One model call for a stage, re-asked until `check` passes: its cost is recorded, and a failure
    marks the stage failed before it is raised again."""
    try:
        draft, call = llm.parse(model, system, prompt, output_type, check=check)
    except LlmError as exc:
        add_llm_cost(assets, run_id, stage, exc.call)
        fail_stage(runs, state, exc, log, what)
        raise
    except Exception as exc:
        fail_stage(runs, state, exc, log, what)
        raise
    add_llm_cost(assets, run_id, stage, call)
    return draft, call


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
        fix, call = llm.parse(model, CORRECTION, correction_request(prompt, brief.moment or brief.description, findings), Correction,
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
