import hashlib
import logging

from src.core.configs import Settings
from src.repository.asset.asset_repository import AssetRepository
from src.repository.run.run_repository import RunRepository
from src.repository.song.song_repository import SongRepository
from src.schema.asset.cost_entry_schema import CostEntry
from src.schema.base_schema import utcnow
from src.schema.enums import StageName, StageStatus
from src.schema.lesson.lesson_schema import BandRules, FormatRules, LessonPlan
from src.schema.run.run_schema import Run
from src.schema.run.stage_state_schema import StageState
from src.schema.song.song_analysis_schema import SongAnalysis
from src.schema.storyboard.storyboard_schema import (CastKind, CastMember, Place, ShotPatch, Slot, Storyboard,
                                                     StoryboardDraft)
from src.service.lesson.lesson_rules import band_rules, format_rules
from src.service.llm.llm_service import LlmCall, LlmError, LlmService
from src.service.storyboard.storyboard_prompt import PROMPT_VERSION, SYSTEM, patch_prompt, user_prompt
from src.service.storyboard.storyboard_rules import (HOST_PLACE, assemble, fix, header_problems, host_for,
                                                     host_place_for, shot_problems, wanted)
from src.service.storyboard.storyboard_slots import make_slots

logger = logging.getLogger(__name__)

# Re-asks per storyboard: the whole board for cast, place and shape problems, then only the failing shots.
# After that the stage fails with the reasons, keeping the draft for inspection.
MAX_HEADER_REASKS = 1
MAX_REASKS = 2


class StoryboardService:
    """Stage 3: from the lesson plan and the song's timing, the shot list the image and video models draw.
    Code cuts the shots on the beat; the model writes one board; code fixes what it can, checks the rest
    and re-asks only the shots that fail. A re-run with the same inputs reuses the stored board."""

    def __init__(self, run_repository: RunRepository, song_repository: SongRepository,
                 asset_repository: AssetRepository, llm: LlmService, settings: Settings) -> None:
        self._runs = run_repository
        self._songs = song_repository
        self._assets = asset_repository
        self._llm = llm
        self._settings = settings

    def run(self, run_id: str, model: str | None = None, force: bool = False) -> Storyboard:
        run = self._run(run_id)
        plan = self._plan(run)
        analysis = self._analysis(run)
        band, fmt = band_rules(plan.grade_band), format_rules(plan.video_type)
        host, host_place = host_for(plan.grade_band), host_place_for(plan.grade_band)
        if [l.index for l in plan.lines] != [l.index for l in analysis.lines]:
            raise ValueError(f"Run {run.id}'s lesson plan is older than its song analysis; run the lesson stage again")
        slots = make_slots(analysis.lines, plan.lines, analysis.beats, analysis.duration_s, band.shot_seconds)
        model = model or run.model_pins.get(StageName.STORYBOARD) or self._settings.storyboard_model
        prompt = user_prompt(plan, analysis.lines, slots, band, fmt, host, host_place)
        key = hashlib.sha256(f"{PROMPT_VERSION}|{model}|{SYSTEM}|{prompt}".encode()).hexdigest()

        state = self._runs.stage(run.id, StageName.STORYBOARD)
        if state is not None and not force and state.status == StageStatus.DONE and state.input_hash == key:
            logger.info("storyboard %s: reused, inputs unchanged", run.id[:8])
            return Storyboard(**state.output)
        logger.info("storyboard %s: %s, grades %s, %s, %d slots (%d repeats), %d sung, %d number", run.id[:8], model,
                    band.band, fmt.type, len(slots), sum(s.repeat_of is not None for s in slots),
                    sum(s.kind == "sung" for s in slots), sum(s.kind == "number" for s in slots))
        state = state or self._runs.stage_states.add(StageState(run_id=run.id, stage=StageName.STORYBOARD))
        state.status, state.attempts, state.error = StageStatus.RUNNING, state.attempts + 1, None
        state.input_hash, state.started_at, state.finished_at = key, utcnow(), None
        state = self._runs.stage_states.update(state)
        spent = 0.0
        try:
            draft, call = self._llm.parse(model, SYSTEM, prompt, StoryboardDraft,
                                          check=lambda d: header_problems(d, band, slots),
                                          max_attempts=1 + MAX_HEADER_REASKS)
            spent += self._log_cost(run.id, call)
            reasks = 0
            problems = self._checked(draft, slots, band, fmt, host, host_place)
            while problems and reasks < MAX_REASKS:
                reasks += 1
                logger.warning("storyboard %s: re-ask %d for %s:\n%s", run.id[:8], reasks, sorted(problems),
                               "\n".join(f"  - {slot_id}: {p}" for slot_id, found in sorted(problems.items())
                                         for p in found))
                patch, call = self._llm.parse(model, SYSTEM, patch_prompt(prompt, draft, problems), ShotPatch,
                                              max_attempts=1)
                spent += self._log_cost(run.id, call)
                logger.info("storyboard %s: re-ask %d returned %s", run.id[:8], reasks,
                            sorted(s.slot for s in patch.shots))
                patched = {s.slot: s for s in patch.shots if s.slot in problems}
                draft.shots = sorted([s for s in draft.shots if s.slot not in problems] + list(patched.values()),
                                     key=lambda s: s.slot)
                problems = self._checked(draft, slots, band, fmt, host, host_place)
            if problems:
                state.output = {"draft": draft.model_dump(mode="json"), "problems": problems}
                raise ValueError(f"storyboard still has problems after {MAX_REASKS} re-asks: " + "; ".join(
                    f"{slot_id}: {', '.join(found)}" for slot_id, found in sorted(problems.items())))
            header = header_problems(draft, band, slots)              # 13: a patch may have moved a shot
            if header:
                raise ValueError("storyboard header broke while fixing shots: " + "; ".join(header))
            used = {s.place for s in draft.shots}
            draft.places = [p for p in draft.places if p.id in used]          # unused places are dropped
            if HOST_PLACE in used:
                draft.places.insert(0, host_place)
            shots, flags = assemble(draft, slots, host)
        except LlmError as exc:
            self._log_cost(run.id, exc.call)
            self._fail(state, exc)
            raise
        except Exception as exc:
            self._fail(state, exc)
            raise
        board = Storyboard(grade_band=plan.grade_band, video_type=plan.video_type, shape=draft.shape,
                           cast=[host] + draft.cast, places=draft.places, motifs=draft.motifs, shots=shots,
                           flags=flags, model=model, prompt_version=PROMPT_VERSION)
        state.status, state.output, state.finished_at = StageStatus.DONE, board.model_dump(mode="json"), utcnow()
        self._runs.stage_states.update(state)
        logger.info("storyboard %s: done, %d shots, %d places, %d cast, $%.4f, %d re-asks, %.0fs", run.id[:8],
                    len(shots), len(board.places), len(board.cast), spent, reasks,
                    (state.finished_at - state.started_at).total_seconds())
        for flag in flags:
            logger.info("storyboard %s: note %s", run.id[:8], flag)
        return board

    def _checked(self, draft: StoryboardDraft, slots: list[Slot], band: BandRules, fmt: FormatRules,
                 host: CastMember, host_place: Place) -> dict[str, list[str]]:
        """Code's fixes, then what is still wrong, per slot."""
        by_id = {s.id: s for s in slots}
        dropped = sorted({s.slot for s in draft.shots} - wanted(slots))
        if dropped:
            logger.warning("storyboard: dropped shots for slots that are not to be written: %s", dropped)
        draft.shots = [s for s in draft.shots if s.slot in wanted(slots)]
        people = {c.id for c in draft.cast if c.kind == CastKind.PERSON}
        for shot in draft.shots:
            for change in fix(shot, by_id[shot.slot], band, people):
                logger.info("storyboard: code fixed %s", change)
        problems = shot_problems(draft, slots, band, fmt, host, host_place)
        logger.info("storyboard: checks found %d problems in %d shots", sum(len(v) for v in problems.values()),
                    len(problems))
        return problems

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

    def _log_cost(self, run_id: str, call: LlmCall) -> float:
        if call.attempts:
            self._assets.cost_entries.add(CostEntry(
                run_id=run_id, stage=StageName.STORYBOARD, provider=call.provider, model=call.model,
                cost_usd=call.cost_usd, units=call.input_tokens + call.output_tokens, unit="tokens"))
        return call.cost_usd

    def _fail(self, state: StageState, exc: Exception) -> None:
        state.status, state.error, state.finished_at = StageStatus.FAILED, str(exc) or type(exc).__name__, utcnow()
        self._runs.stage_states.update(state)
        logger.error("storyboard %s: failed: %s", state.run_id[:8], state.error)
