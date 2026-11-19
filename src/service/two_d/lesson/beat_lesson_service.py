import hashlib
import logging

from src.core.configs import Settings
from src.repository.asset.asset_repository import AssetRepository
from src.repository.run.run_repository import RunRepository
from src.repository.song.song_repository import SongRepository
from src.schema.base_schema import utcnow
from src.schema.enums import StageName, StageStatus
from src.schema.lesson.lesson_schema import GradeBand
from src.schema.two_d.beat_schema import BeatLessonDraft, BeatLessonPlan
from src.service.two_d.lesson.beat_prompt import PROMPT_VERSION, SYSTEM, jamwise_direction, user_prompt
from src.service.two_d.lesson.beat_rules import problems, timed_beats
from src.service.llm.llm_service import LlmService
from src.service.stage.stage_common import analysis_of, export_of, parse_stage, require_film_type, run_of, start_stage
from src.service.two_d.common import TWO_D

logger = logging.getLogger(__name__)


class BeatLessonService:
    """Stage 2: from the timed lyrics, the teaching beats: what each part of the song teaches, its key word
    and when it is sung. The model drafts; code checks the draft against the rules, re-asks until it passes,
    and sets the beats in time from the lyric timing. A re-run with the same inputs reuses the stored plan
    instead of paying again."""

    def __init__(self, run_repository: RunRepository, song_repository: SongRepository,
                 asset_repository: AssetRepository, llm: LlmService, settings: Settings) -> None:
        self._runs = run_repository
        self._songs = song_repository
        self._assets = asset_repository
        self._llm = llm
        self._settings = settings

    def run(self, run_id: str, model: str | None = None, force: bool = False) -> BeatLessonPlan:
        run = run_of(self._runs, run_id)
        require_film_type(self._runs, run.id, TWO_D)
        if run.audience is None:
            raise ValueError(f"Run {run.id} has no grade band; create it with --grade")
        band = GradeBand(run.audience)
        analysis = analysis_of(self._songs, run)
        lines = analysis.lines
        if not lines:
            raise ValueError(f"Run {run.id}'s song has no lyric lines")
        export = export_of(self._songs, run)
        model = model or run.model_pins.get(StageName.LESSON) or self._settings.lesson_model
        prompt = user_prompt(lines, band, run.brief, jamwise_direction(export) if export else None)
        key = hashlib.sha256(f"{PROMPT_VERSION}|{model}|{SYSTEM}|{prompt}".encode()).hexdigest()

        state = self._runs.stage(run.id, StageName.LESSON)
        if state is not None and not force and state.status == StageStatus.DONE and state.input_hash == key:
            logger.info("lesson %s: reused, inputs unchanged", run.id[:8])
            return BeatLessonPlan(**state.output)
        logger.info("lesson %s: %s, grades %s, %d lines", run.id[:8], model, band, len(lines))
        state = start_stage(self._runs, state, run.id, StageName.LESSON, key)
        draft, call = parse_stage(self._llm, self._assets, self._runs, state, run.id, StageName.LESSON, "lesson",
                                  logger, model, SYSTEM, prompt, BeatLessonDraft,
                                  lambda d: problems(d, lines, band, analysis.duration_s))
        plan = BeatLessonPlan(**draft.model_dump(exclude={"beats"}),
                          beats=timed_beats(draft.beats, lines, analysis.words, analysis.duration_s),
                          grade_band=band, model=model, prompt_version=PROMPT_VERSION)
        state.status, state.output, state.finished_at = StageStatus.DONE, plan.model_dump(mode="json"), utcnow()
        self._runs.stage_states.update(state)
        logger.info("lesson %s: done, %d beats, $%.4f in %d attempts, %.0fs", run.id[:8], len(plan.beats),
                    call.cost_usd, call.attempts, (state.finished_at - state.started_at).total_seconds())
        return plan
