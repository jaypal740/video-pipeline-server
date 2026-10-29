import hashlib
import logging

from src.core.configs import Settings
from src.repository.asset.asset_repository import AssetRepository
from src.repository.run.run_repository import RunRepository
from src.repository.song.song_repository import SongRepository
from src.schema.asset.cost_entry_schema import CostEntry
from src.schema.base_schema import utcnow
from src.schema.enums import StageName, StageStatus
from src.schema.lesson.lesson_schema import GradeBand, LessonDraft, LessonPlan, LineTag, VideoType
from src.schema.run.run_schema import Run
from src.schema.run.stage_state_schema import StageState
from src.schema.song.song_analysis_schema import LyricLine
from src.service.lesson.lesson_prompt import PROMPT_VERSION, SYSTEM, user_prompt
from src.service.lesson.lesson_rules import allowed_types, problems
from src.service.llm.llm_service import LlmCall, LlmError, LlmService

logger = logging.getLogger(__name__)


class LessonService:
    """Stage 2: from the timed lyrics, what each line teaches and shows, and which video type fits the
    grade band. The model drafts; code checks the draft against the band and type rules and re-asks until
    it passes. A re-run with the same inputs reuses the stored plan instead of paying again."""

    def __init__(self, run_repository: RunRepository, song_repository: SongRepository,
                 asset_repository: AssetRepository, llm: LlmService, settings: Settings) -> None:
        self._runs = run_repository
        self._songs = song_repository
        self._assets = asset_repository
        self._llm = llm
        self._settings = settings

    def run(self, run_id: str, model: str | None = None, video_type: VideoType | None = None,
            force: bool = False) -> LessonPlan:
        run = self._run(run_id)
        if run.audience is None:
            raise ValueError(f"Run {run.id} has no grade band; create it with --grade")
        band = GradeBand(run.audience)
        types = allowed_types(band, video_type)
        lines = self._lines(run)
        model = model or run.model_pins.get(StageName.LESSON) or self._settings.lesson_model
        prompt = user_prompt(lines, band, types, run.brief)
        key = hashlib.sha256(f"{PROMPT_VERSION}|{model}|{SYSTEM}|{prompt}".encode()).hexdigest()

        state = self._runs.stage(run.id, StageName.LESSON)
        if state is not None and not force and state.status == StageStatus.DONE and state.input_hash == key:
            logger.info("lesson %s: reused, inputs unchanged", run.id[:8])
            return LessonPlan(**state.output)
        logger.info("lesson %s: %s, grades %s, %d lines, types %s", run.id[:8], model, band, len(lines),
                    [str(t) for t in types])
        state = state or self._runs.stage_states.add(StageState(run_id=run.id, stage=StageName.LESSON))
        state.status, state.attempts, state.error = StageStatus.RUNNING, state.attempts + 1, None
        state.input_hash, state.started_at, state.finished_at = key, utcnow(), None
        state = self._runs.stage_states.update(state)
        try:
            draft, call = self._llm.parse(model, SYSTEM, prompt, LessonDraft,
                                          check=lambda d: problems(d, lines, band, types))
        except LlmError as exc:
            self._log_cost(run.id, exc.call)
            self._fail(state, exc)
            raise
        except Exception as exc:
            self._fail(state, exc)
            raise
        self._log_cost(run.id, call)
        unclear = {l.index for l in lines if l.unclear}
        for entry in draft.lines:                         # intake decides what is unclear, not the model
            if entry.index in unclear:
                entry.tag, entry.number_text = LineTag.UNCLEAR, None
        plan = LessonPlan(**draft.model_dump(), grade_band=band, model=model, prompt_version=PROMPT_VERSION)
        state.status, state.output, state.finished_at = StageStatus.DONE, plan.model_dump(mode="json"), utcnow()
        self._runs.stage_states.update(state)
        logger.info("lesson %s: done, %s, %d lines, $%.4f in %d attempts, %.0fs", run.id[:8], plan.video_type,
                    len(plan.lines), call.cost_usd, call.attempts, (state.finished_at - state.started_at).total_seconds())
        return plan

    def _run(self, run_id: str) -> Run:
        run = self._runs.runs.get(run_id)
        if run is None:
            raise LookupError(f"Run {run_id} not found")
        return run

    def _lines(self, run: Run) -> list[LyricLine]:
        song = self._songs.songs.get(run.song_id)
        if song is None or song.analysis_id is None:
            raise LookupError(f"Run {run.id}'s song has not been analysed; run intake on it first")
        return self._songs.analyses.get(song.analysis_id).lines

    def _log_cost(self, run_id: str, call: LlmCall) -> None:
        if call.attempts:
            self._assets.cost_entries.add(CostEntry(
                run_id=run_id, stage=StageName.LESSON, provider=call.provider, model=call.model,
                cost_usd=call.cost_usd, units=call.input_tokens + call.output_tokens, unit="tokens"))

    def _fail(self, state: StageState, exc: Exception) -> None:
        state.status, state.error, state.finished_at = StageStatus.FAILED, str(exc) or type(exc).__name__, utcnow()
        self._runs.stage_states.update(state)
        logger.error("lesson %s: failed: %s", state.run_id[:8], state.error)
