import hashlib
import logging

from src.core.configs import Settings
from src.repository.asset.asset_repository import AssetRepository
from src.repository.run.run_repository import RunRepository
from src.repository.song.song_repository import SongRepository
from src.schema.base_schema import utcnow
from src.schema.enums import FilmType, StageName, StageStatus
from src.schema.lesson.lesson_schema import GradeBand, LessonDraft, LessonPlan, LineTag, VideoType
from src.service.lesson.lesson_prompt import (JAMWISE_SYSTEM, PROMPT_VERSION, SYSTEM, featured_terms,
                                              jamwise_direction, scene_ids, sequence_limit, user_prompt)
from src.service.lesson.lesson_rules import (MAX_SEQUENCES, allowed_types, chorus_lines, clear_keywords, problems,
                                             scene_notes, sing_the_chorus)
from src.service.llm.llm_service import LlmService
from src.service.stage.stage_common import (REALISTIC, analysis_of, export_of, film_type_of, parse_stage, require_film_type,
                                            run_of, start_stage)

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
            film_type: FilmType | None = None, force: bool = False) -> LessonPlan:
        run = run_of(self._runs, run_id)
        require_film_type(self._runs, run.id, REALISTIC)
        if run.audience is None:
            raise ValueError(f"Run {run.id} has no grade band; create it with --grade")
        band = GradeBand(run.audience)
        types = allowed_types(band, video_type)
        lines = analysis_of(self._songs, run).lines
        export = export_of(self._songs, run)
        film_type = (film_type or film_type_of(self._runs, run.id)
                     or (FilmType.MUSIC_VIDEO if export else FilmType.NARRATION))
        if film_type not in REALISTIC:
            raise ValueError(f"A {film_type} film's lesson is made with `run 2d lesson`")
        if film_type == FilmType.MUSIC_VIDEO and export is None:
            raise ValueError(f"Run {run.id[:8]}: music_video needs a JamWise song; this song has no JamWise export")
        if film_type == FilmType.NARRATION:
            export = None                                 # a narration film follows its lyrics, not a JamWise plan
        featured = featured_terms(export) if export else []
        most = sequence_limit(export) if export else MAX_SEQUENCES
        chorus = chorus_lines(lines) if export else set()
        system = JAMWISE_SYSTEM if export else SYSTEM
        model = model or run.model_pins.get(StageName.LESSON) or self._settings.lesson_model
        prompt = user_prompt(lines, band, types, run.brief, jamwise_direction(export) if export else None)
        key = hashlib.sha256(f"{PROMPT_VERSION}|{model}|{system}|{prompt}".encode()).hexdigest()

        state = self._runs.stage(run.id, StageName.LESSON)
        if state is not None and not force and state.status == StageStatus.DONE and state.input_hash == key:
            logger.info("lesson %s: reused, inputs unchanged, %s", run.id[:8], film_type)
            plan = LessonPlan(**state.output)
            if plan.film_type != film_type:                 # same lesson, its type set or changed: no new call
                plan.film_type, state.output = film_type, {**state.output, "film_type": str(film_type)}
                self._runs.stage_states.update(state)
            return plan
        logger.info("lesson %s: %s, %s, grades %s, %d lines, types %s", run.id[:8], film_type, model, band, len(lines),
                    [str(t) for t in types])
        state = start_stage(self._runs, state, run.id, StageName.LESSON, key)
        draft, call = parse_stage(self._llm, self._assets, self._runs, state, run.id, StageName.LESSON, "lesson",
                                  logger, model, system, prompt, LessonDraft,
                                  lambda d: problems(d, lines, band, types, featured, most, bool(chorus)))
        if chorus:
            sing_the_chorus(draft, chorus)
        unclear = {l.index for l in lines if l.unclear}
        for entry in draft.lines:                         # intake decides what is unclear, not the model
            if entry.index in unclear:
                entry.tag, entry.number_text = LineTag.UNCLEAR, None
        for note in clear_keywords(draft, lines) + (scene_notes(draft, scene_ids(export)) if export else []):
            logger.warning("lesson %s: %s", run.id[:8], note)
        plan = LessonPlan(**draft.model_dump(), grade_band=band, film_type=film_type, model=model,
                          prompt_version=PROMPT_VERSION)
        state.status, state.output, state.finished_at = StageStatus.DONE, plan.model_dump(mode="json"), utcnow()
        self._runs.stage_states.update(state)
        logger.info("lesson %s: done, %s, %s, %d lines, $%.4f in %d attempts, %.0fs", run.id[:8], film_type,
                    plan.video_type, len(plan.lines), call.cost_usd, call.attempts,
                    (state.finished_at - state.started_at).total_seconds())
        return plan
