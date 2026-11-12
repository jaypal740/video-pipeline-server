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
from src.schema.storyboard.script_schema import (AnimationScript, BeatScript, BeatScriptDraft, DrawingList, Screen,
                                                 ScriptPatch, ScriptRepair, ScriptReview)
from src.service.intake.intake_jamwise import JAMWISE_FILE, read_export
from src.service.llm.llm_service import LlmError, LlmService
from src.service.stage.stage_common import add_llm_cost, fail_stage
from src.service.storyboard.script_prompt import (PROMPT_VERSION, SYSTEM, drawings_prompt, jamwise_performer,
                                                  jamwise_pictures, repair_prompt, review_prompt, script_prompt,
                                                  song_prompt)
from src.service.storyboard.script_rules import (BY_CODE, cover_problems, drawing_problems, finish, mend, play,
                                                 repair_problems, style_of)

logger = logging.getLogger(__name__)

# The model is asked again this many times, for the drawings and for each run of beats, when code's checks
# fail. After that the stage fails with the reasons, keeping what was written for inspection.
MAX_REASKS = 2
# Beats scripted per call: few enough for the model to time every action, enough to see how beats join.
BEATS_PER_CALL = 6
# A finished script is reviewed against the plan. If the review finds gaps, only the beats it names are
# repaired, once, and the script is reviewed again: two tries in all. The script with the fewer gaps is kept,
# the repaired one when they tie, and the stage carries on with it: a gap left over is recorded, not fatal.
REPAIRS = 1


class StoryboardService:
    """Stage 3: from the lesson's beats, the animation script the drawings and the render are made from.
    The model first lists every drawing the video needs, each made once and reused; then it scripts the
    beats a few at a time, saying what is on screen and what happens when. Code checks each answer, lays it
    out to see that no drawing hides another, re-asks until it passes, and adds the key words, figures and
    lip-sync from the lesson. The finished script is
    then reviewed against the plan, before any drawing is paid for, and the beats that leave gaps are repaired.
    A re-run with the same inputs reuses the stored script."""

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
        model = model or run.model_pins.get(StageName.STORYBOARD) or self._settings.storyboard_model
        song = self._song(run, plan, analysis)
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
        spent, reasks, listed, scripts, screens, kept = 0.0, 0, None, [], [], None
        terms = [t.term for t in plan.key_terms]
        wide = self._settings.assembly_width / self._settings.assembly_height

        def laid(patch: ScriptPatch, beats: list, before: Screen) -> list[str]:
            """What is wrong with a run of beats: code's rules, then what hides what once it is laid out."""
            found, after = play(patch.beats, beats, drawings, before)
            return found or cover_problems(listed, finish(
                scripts + patch.beats, plan.beats[:len(scripts) + len(beats)], screens + after, analysis.words,
                analysis.duration_s, terms), wide, beats[0].start)

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
                                              ScriptPatch, check=lambda p: laid(p, beats, before),
                                              max_attempts=1 + MAX_REASKS)
                spent += add_llm_cost(self._assets, run.id, StageName.STORYBOARD, call)
                reasks += call.attempts - 1
                scripts += patch.beats
                screens += play(patch.beats, beats, drawings, before)[1]
                logger.info("storyboard %s: beats %s to %s, %d attempts", run.id[:8], beats[0].id, beats[-1].id,
                            call.attempts)
            for n in range(1 + REPAIRS):
                done = finish(scripts, plan.beats, screens, analysis.words, analysis.duration_s, terms)
                review, call = self._llm.parse(model, SYSTEM, review_prompt(song, listed, done), ScriptReview)
                spent += add_llm_cost(self._assets, run.id, StageName.STORYBOARD, call)
                logger.info("storyboard %s: review %d, %d gaps%s", run.id[:8], n + 1, len(review.gaps),
                            "".join(f"\n  - {g}" for g in review.gaps))
                if kept is None or len(review.gaps) <= len(kept[0]):
                    kept = (review.gaps, listed, scripts, done)
                gaps, listed, scripts, done = kept                      # a repair that made things worse is dropped
                if not gaps or n == REPAIRS:
                    break
                try:
                    repair, call = self._llm.parse(
                        model, SYSTEM, repair_prompt(song, listed, scripts, gaps), ScriptRepair,
                        check=lambda r: self._mended(r, listed, scripts, plan, analysis)[0],
                        max_attempts=1 + MAX_REASKS)
                except LlmError as exc:                                 # no usable repair: carry on with the best script
                    spent += add_llm_cost(self._assets, run.id, StageName.STORYBOARD, exc.call)
                    logger.warning("storyboard %s: repair %d gave nothing usable: %s", run.id[:8], n + 1, exc)
                    break
                spent += add_llm_cost(self._assets, run.id, StageName.STORYBOARD, call)
                reasks += call.attempts - 1
                listed, scripts = mend(listed, scripts, repair)
                screens = play(scripts, plan.beats, {d.id: d for d in listed.drawings}, Screen())[1]
                logger.info("storyboard %s: repair %d, beats %s, %d new drawings, %d attempts", run.id[:8], n + 1,
                            " ".join(b.beat for b in repair.beats), len(repair.drawings), call.attempts)
        except LlmError as exc:
            add_llm_cost(self._assets, run.id, StageName.STORYBOARD, exc.call)
            state.output = {"drawings": listed.model_dump(mode="json") if listed else None,
                            "scripts": [s.model_dump(mode="json") for s in scripts]}
            fail_stage(self._runs, state, exc, logger, "storyboard")
            raise
        except Exception as exc:
            fail_stage(self._runs, state, exc, logger, "storyboard")
            raise
        gaps, listed, _, done = kept
        board = self._board(plan, listed, done, gaps, model)
        state.status, state.output, state.finished_at = StageStatus.DONE, board.model_dump(mode="json"), utcnow()
        self._runs.stage_states.update(state)
        logger.info("storyboard %s: done, %d beats, %d of %d drawings used, %d gaps left, $%.4f, %d re-asks, %.0fs",
                    run.id[:8], len(board.beats), len(board.drawings), len(listed.drawings), len(gaps), spent, reasks,
                    (state.finished_at - state.started_at).total_seconds())
        return board

    def repair(self, run_id: str, gaps: list[str]) -> AnimationScript:
        """The stored script with the beats `gaps` name written again: what a later check found wrong in the
        film is closed the way the script's own review closes a gap, and checked by code the same way. The
        stage's inputs have not changed, so a re-run of it still reuses the script."""
        run = self._run(run_id)
        plan, analysis = self._plan(run), self._analysis(run)
        state = self._runs.stage(run.id, StageName.STORYBOARD)
        if state is None or state.status != StageStatus.DONE:
            raise ValueError(f"Run {run.id} has no animation script; run the storyboard stage first")
        board = AnimationScript(**state.output)
        listed = DrawingList(performer=board.performer, palette=board.palette, drawings=board.drawings)
        scripts = [BeatScriptDraft(beat=b.beat, background=b.background,
                                   actions=[a for a in b.actions if a.verb not in BY_CODE]) for b in board.beats]
        try:
            repair, call = self._llm.parse(
                board.model, SYSTEM, repair_prompt(self._song(run, plan, analysis), listed, scripts, gaps), ScriptRepair,
                check=lambda r: self._mended(r, listed, scripts, plan, analysis)[0], max_attempts=1 + MAX_REASKS)
        except LlmError as exc:
            add_llm_cost(self._assets, run.id, StageName.STORYBOARD, exc.call)
            raise
        add_llm_cost(self._assets, run.id, StageName.STORYBOARD, call)
        done = self._mended(repair, listed, scripts, plan, analysis)[1]
        board = self._board(plan, mend(listed, scripts, repair)[0], done, board.gaps, board.model)
        self.restore(run.id, board)
        logger.info("storyboard %s: repaired beats %s for %d gaps, %d new drawings, $%.4f", run.id[:8],
                    " ".join(b.beat for b in repair.beats), len(gaps), len(repair.drawings), call.cost_usd)
        return board

    def restore(self, run_id: str, board: AnimationScript) -> None:
        """Make `board` the run's stored script."""
        state = self._runs.stage(run_id, StageName.STORYBOARD)
        state.output = board.model_dump(mode="json")
        self._runs.stage_states.update(state)

    def _mended(self, repair: ScriptRepair, listed: DrawingList, scripts: list[BeatScriptDraft], plan: LessonPlan,
                analysis: SongAnalysis) -> tuple[list[str], list[BeatScript]]:
        """What is wrong with a repair, by code's rules and then by what hides what once the whole script is
        laid out with it; and that script, finished."""
        found = repair_problems(repair, listed, scripts, plan.beats)
        if found:
            return found, []
        new, fixed = mend(listed, scripts, repair)
        after = play(fixed, plan.beats, {d.id: d for d in new.drawings}, Screen())[1]
        done = finish(fixed, plan.beats, after, analysis.words, analysis.duration_s, [t.term for t in plan.key_terms])
        return cover_problems(new, done, self._settings.assembly_width / self._settings.assembly_height), done

    def _song(self, run: Run, plan: LessonPlan, analysis: SongAnalysis) -> str:
        export = self._jamwise(run)
        return song_prompt(plan, analysis.lines, jamwise_performer(export) if export else None,
                           jamwise_pictures(export) if export else None)

    @staticmethod
    def _board(plan: LessonPlan, listed: DrawingList, done: list[BeatScript], gaps: list[str], model: str) -> AnimationScript:
        """The finished script, with only the drawings its beats use and those they are drawn from."""
        used = {s.background for s in done} | {i for s in done for a in s.actions for i in (a.drawing, a.to)}
        used |= {d.of for d in listed.drawings if d.id in used}
        return AnimationScript(
            grade_band=plan.grade_band, style=style_of(plan.grade_band), palette=listed.palette,
            performer=listed.performer, drawings=[d for d in listed.drawings if d.id in used],
            beats=done, gaps=gaps, model=model, prompt_version=PROMPT_VERSION)

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
