import hashlib
import logging

from src.core.configs import Settings
from src.repository.asset.asset_repository import AssetRepository
from src.repository.run.run_repository import RunRepository
from src.repository.song.song_repository import SongRepository
from src.schema.base_schema import utcnow
from src.schema.enums import StageName, StageStatus
from src.schema.lesson.lesson_schema import BandRules, FormatRules, LessonPlan
from src.schema.run.run_schema import Run
from src.schema.run.stage_state_schema import StageState
from src.schema.song.song_analysis_schema import SongAnalysis
from src.schema.storyboard.storyboard_schema import (BoardHeader, CastKind, CastMember, SequencePlaces, ShotDraft,
                                                     ShotPatch, Slot, Storyboard, StoryboardDraft)
from src.service.lesson.lesson_rules import band_rules, format_rules
from src.service.llm.llm_service import LlmError, LlmService
from src.service.stage.stage_common import add_llm_cost, fail_stage
from src.service.storyboard.storyboard_prompt import (PROMPT_VERSION, SYSTEM, header_prompt, sequence_prompt,
                                                      user_prompt)
from src.service.storyboard.storyboard_rules import (assemble, fix, header_problems, host_for,
                                                     sequence_place_problems, shot_problems, wanted, wardrobe_for,
                                                     zoom_only)
from src.service.storyboard.storyboard_slots import make_slots

logger = logging.getLogger(__name__)

# Re-asks: the header for cast, place and shape problems, then a whole sequence for any of its shots'
# problems. After that the stage fails with the reasons, keeping the draft for inspection.
MAX_HEADER_REASKS = 1
MAX_REASKS = 2


class StoryboardService:
    """Stage 3: from the lesson plan and the song's timing, the shot list the image and video models draw.
    Code cuts the shots on the beat; the model writes the header for the whole video, then the shots one
    lesson sequence at a time, each with the whole film in view; code fixes what it can, checks the rest
    and re-asks a failing sequence whole. A re-run with the same inputs reuses the stored board."""

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
        host, wardrobe = host_for(plan.grade_band), wardrobe_for(plan.grade_band)
        if [l.index for l in plan.lines] != [l.index for l in analysis.lines]:
            raise ValueError(f"Run {run.id}'s lesson plan is older than its song analysis; run the lesson stage again")
        if not plan.sequences:
            raise ValueError(f"Run {run.id}'s lesson plan has no sequences; run the lesson stage again")
        slots = make_slots(analysis.lines, plan.lines, analysis.beats, analysis.duration_s, band.shot_seconds,
                           plan.sequences)
        model = model or run.model_pins.get(StageName.STORYBOARD) or self._settings.storyboard_model
        prompt = user_prompt(plan, analysis.lines, slots, band, fmt, host, wardrobe)
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
        spent, reasks, done = 0.0, 0, []
        try:
            header, call = self._llm.parse(
                model, SYSTEM, header_prompt(prompt, plan.sequences), BoardHeader,
                check=lambda h: (header_problems(_draft(h, []), band, slots, zoom_only(h.sequence_places))
                                 + sequence_place_problems(h, plan.sequences)),
                max_attempts=1 + MAX_HEADER_REASKS)
            spent += add_llm_cost(self._assets, run.id, StageName.STORYBOARD, call)
            for k in range(len(plan.sequences)):
                todo = sorted(wanted([s for s in slots if s.sequence == k]))
                if not todo:
                    logger.info("storyboard %s: sequence %d is all repeats, copied", run.id[:8], k + 1)
                    continue
                upto = [s for s in slots if s.sequence <= k]
                check = lambda patch: self._sequence_problems(patch, header, done, upto, set(todo), band, fmt,
                                                              host)
                try:
                    patch, call = self._llm.parse(model, SYSTEM, sequence_prompt(prompt, header, done, k,
                                                                                  plan.sequences, todo),
                                                  ShotPatch, check=check, max_attempts=1 + MAX_REASKS)
                except LlmError:
                    state.output = {"draft": _draft(header, done).model_dump(mode="json"), "sequence": k + 1}
                    raise
                spent += add_llm_cost(self._assets, run.id, StageName.STORYBOARD, call)
                reasks += call.attempts - 1
                done += patch.shots
                logger.info("storyboard %s: sequence %d of %d, %d shots, %d attempts", run.id[:8], k + 1,
                            len(plan.sequences), len(patch.shots), call.attempts)
            draft = _draft(header, done)
            problems = header_problems(draft, band, slots, zoom_only(header.sequence_places))
            if problems:
                state.output = {"draft": draft.model_dump(mode="json"), "problems": problems}
                raise ValueError("storyboard header broke once every sequence was written: " + "; ".join(problems))
            used = {s.place for s in draft.shots}
            draft.places = [p for p in draft.places if p.id in used]          # unused places are dropped
            shots, flags = assemble(draft, slots, host)
        except LlmError as exc:
            add_llm_cost(self._assets, run.id, StageName.STORYBOARD, exc.call)
            fail_stage(self._runs, state, exc, logger, "storyboard")
            raise
        except Exception as exc:
            fail_stage(self._runs, state, exc, logger, "storyboard")
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

    def _sequence_problems(self, patch: ShotPatch, header: BoardHeader, done: list[ShotDraft], upto: list[Slot],
                           todo: set[str], band: BandRules, fmt: FormatRules, host: CastMember) -> list[str]:
        """One sequence's shots checked after the shots before them, so the join between sequences is checked
        too. Shots for slots outside the sequence are dropped."""
        patch.shots = [s for s in patch.shots if s.slot in todo]
        found = self._checked(_draft(header, done + patch.shots), upto, band, fmt, host, header.sequence_places)
        return [f"{slot_id}: {p}" for slot_id, problems in sorted(found.items()) if slot_id in todo for p in problems]

    def _checked(self, draft: StoryboardDraft, slots: list[Slot], band: BandRules, fmt: FormatRules,
                 host: CastMember, sequence_places: list[SequencePlaces]) -> dict[str, list[str]]:
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
        problems = shot_problems(draft, slots, band, fmt, host, sequence_places)
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


def _draft(header: BoardHeader, shots: list[ShotDraft]) -> StoryboardDraft:
    return StoryboardDraft(**header.model_dump(exclude={"sequence_places"}), shots=shots)
