import hashlib
import logging
from pathlib import Path

from src.core.configs import Settings
from src.repository.asset.asset_repository import AssetRepository
from src.repository.run.run_repository import RunRepository
from src.repository.song.song_repository import SongRepository
from src.schema.base_schema import utcnow
from src.schema.enums import StageName, StageStatus
from src.schema.lesson.lesson_schema import LessonPlan
from src.schema.run.run_schema import Run
from src.schema.run.stage_state_schema import StageState
from src.schema.song.song_analysis_schema import SongAnalysis
from src.schema.storyboard.script_schema import AnimationScript, DrawingList, Screen, ScriptPatch
from src.service.intake.intake_jamwise import JAMWISE_FILE, read_export
from src.service.llm.llm_service import LlmError, LlmService
from src.service.stage.stage_common import add_llm_cost, fail_stage
from src.service.storyboard.script_prompt import (PROMPT_VERSION, SYSTEM, drawings_prompt, jamwise_performer,
                                                  script_prompt, song_prompt)
from src.service.storyboard.script_rules import drawing_problems, finish, play, style_of

logger = logging.getLogger(__name__)

# The model is asked again this many times, for the drawings and for each run of beats, when code's checks
# fail. After that the stage fails with the reasons, keeping what was written for inspection.
MAX_REASKS = 2
# Beats scripted per call: few enough for the model to time every action, enough to see how beats join.
BEATS_PER_CALL = 6


class StoryboardService:
    """Stage 3: from the lesson's beats, the animation script the drawings and the render are made from.
    The model first lists every drawing the video needs, each made once and reused; then it scripts the
    beats a few at a time, saying what is on screen and what happens when. Code checks each answer, re-asks
    until it passes, and adds the key words, figures and lip-sync from the lesson. A re-run with the same
    inputs reuses the stored script."""

    def __init__(self, run_repository: RunRepository, song_repository: SongRepository,
                 asset_repository: AssetRepository, llm: LlmService, settings: Settings) -> None:
        self._runs = run_repository
        self._songs = song_repository
        self._assets = asset_repository
        self._llm = llm
        self._settings = settings

    def run(self, run_id: str, model: str | None = None, force: bool = False) -> AnimationScript:
        run = self._run(run_id)
        plan = self._plan(run)
        analysis = self._analysis(run)
        if [i for b in plan.beats for i in b.lines] != [l.index for l in analysis.lines]:
            raise ValueError(f"Run {run.id}'s lesson plan is older than its song analysis; run the lesson stage again")
        export = self._jamwise(run)
        model = model or run.model_pins.get(StageName.STORYBOARD) or self._settings.storyboard_model
        song = song_prompt(plan, analysis.lines, jamwise_performer(export) if export else None)
        ask = drawings_prompt(song)
        key = hashlib.sha256(f"{PROMPT_VERSION}|{model}|{SYSTEM}|{ask}".encode()).hexdigest()

        state = self._runs.stage(run.id, StageName.STORYBOARD)
        if state is not None and not force and state.status == StageStatus.DONE and state.input_hash == key:
            logger.info("storyboard %s: reused, inputs unchanged", run.id[:8])
            return AnimationScript(**state.output)
        logger.info("storyboard %s: %s, grades %s, %d beats", run.id[:8], model, plan.grade_band, len(plan.beats))
        state = state or self._runs.stage_states.add(StageState(run_id=run.id, stage=StageName.STORYBOARD))
        state.status, state.attempts, state.error = StageStatus.RUNNING, state.attempts + 1, None
        state.input_hash, state.started_at, state.finished_at = key, utcnow(), None
        state = self._runs.stage_states.update(state)
        spent, reasks, listed, scripts, screens = 0.0, 0, None, [], []
        try:
            listed, call = self._llm.parse(model, SYSTEM, ask, DrawingList,
                                           check=lambda d: drawing_problems(d, plan.beats),
                                           max_attempts=1 + MAX_REASKS)
            spent += add_llm_cost(self._assets, run.id, StageName.STORYBOARD, call)
            reasks += call.attempts - 1
            drawings = {d.id: d for d in listed.drawings}
            logger.info("storyboard %s: %d drawings, %d attempts", run.id[:8], len(drawings), call.attempts)
            for i in range(0, len(plan.beats), BEATS_PER_CALL):
                beats = plan.beats[i:i + BEATS_PER_CALL]
                before = screens[-1] if screens else Screen()
                patch, call = self._llm.parse(model, SYSTEM, script_prompt(song, listed, scripts, beats, before),
                                              ScriptPatch, check=lambda p: play(p.beats, beats, drawings, before)[0],
                                              max_attempts=1 + MAX_REASKS)
                spent += add_llm_cost(self._assets, run.id, StageName.STORYBOARD, call)
                reasks += call.attempts - 1
                scripts += patch.beats
                screens += play(patch.beats, beats, drawings, before)[1]
                logger.info("storyboard %s: beats %s to %s, %d attempts", run.id[:8], beats[0].id, beats[-1].id,
                            call.attempts)
        except LlmError as exc:
            add_llm_cost(self._assets, run.id, StageName.STORYBOARD, exc.call)
            state.output = {"drawings": listed.model_dump(mode="json") if listed else None,
                            "scripts": [s.model_dump(mode="json") for s in scripts]}
            fail_stage(self._runs, state, exc, logger, "storyboard")
            raise
        except Exception as exc:
            fail_stage(self._runs, state, exc, logger, "storyboard")
            raise
        used = {s.background for s in scripts} | {i for s in scripts for a in s.actions for i in (a.drawing, a.to)}
        used |= {d.of for d in listed.drawings if d.id in used}         # what a used drawing is drawn from stays
        board = AnimationScript(
            grade_band=plan.grade_band, style=style_of(plan.grade_band), palette=listed.palette,
            performer=listed.performer, drawings=[d for d in listed.drawings if d.id in used],
            beats=finish(scripts, plan.beats, screens, analysis.words, analysis.duration_s,
                         [t.term for t in plan.key_terms]),
            model=model, prompt_version=PROMPT_VERSION)
        state.status, state.output, state.finished_at = StageStatus.DONE, board.model_dump(mode="json"), utcnow()
        self._runs.stage_states.update(state)
        logger.info("storyboard %s: done, %d beats, %d of %d drawings used, $%.4f, %d re-asks, %.0fs", run.id[:8],
                    len(board.beats), len(board.drawings), len(drawings), spent, reasks,
                    (state.finished_at - state.started_at).total_seconds())
        return board

    def _run(self, run_id: str) -> Run:
        run = self._runs.runs.get(run_id)
        if run is None:
            raise LookupError(f"Run {run_id} not found")
        return run

    def _plan(self, run: Run) -> LessonPlan:
        state = self._runs.stage(run.id, StageName.LESSON)
        if state is None or state.status != StageStatus.DONE:
            raise ValueError(f"Run {run.id} has no lesson plan; run the lesson stage first")
        return LessonPlan(**state.output)

    def _analysis(self, run: Run) -> SongAnalysis:
        song = self._songs.songs.get(run.song_id)
        if song is None or song.analysis_id is None:
            raise LookupError(f"Run {run.id}'s song has not been analysed; run intake on it first")
        return self._songs.analyses.get(song.analysis_id)

    def _jamwise(self, run: Run) -> dict | None:
        """The song's JamWise export, if it came from JamWise."""
        export = Path(self._songs.songs.get(run.song_id).audio_path).parent / JAMWISE_FILE
        return read_export(export) if export.is_file() else None
