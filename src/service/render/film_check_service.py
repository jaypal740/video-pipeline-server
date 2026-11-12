import hashlib
import io
import json
import logging

from src.core.configs import Settings
from src.repository.asset.asset_repository import AssetRepository
from src.repository.run.run_repository import RunRepository
from src.repository.song.song_repository import SongRepository
from src.schema.base_schema import utcnow
from src.schema.enums import StageName, StageStatus
from src.schema.lesson.lesson_schema import LessonPlan
from src.schema.run.stage_state_schema import StageState
from src.schema.storyboard.script_schema import AnimationScript, FilmProblem, FilmReview, Verb
from src.service.drawing.drawing_service import DrawingService
from src.service.llm.llm_service import LabelledImage, LlmError, LlmService
from src.service.render.render_service import RenderService
from src.service.render.timeline import scenes_of
from src.service.stage.stage_common import add_llm_cost, fail_stage, script_of
from src.service.storyboard.script_prompt import _action
from src.service.storyboard.storyboard_service import StoryboardService

logger = logging.getLogger(__name__)

STAGE = StageName.QC
VERSION = "filmcheck.v3"    # part of the check's inputs: raise it when the same film should be judged again
# The judge sees the film as at most MAX_FRAMES frames, each HEIGHT pixels tall: every drawing once it has
# come, SETTLE seconds after it or anything else began to fade in or out, and, where there is room, the
# picture after each move, zoom and key word, at least APART seconds from the frames already chosen.
MAX_FRAMES, HEIGHT, SETTLE, APART = 90, 480, 0.4, 0.75
# When the judge finds problems the script's beats they name are written again, this many times.
REPAIRS = 3

SYSTEM = """You check an educational music video for school children before it is finished. It is flat 2D \
motion graphics: drawings appear, move and swap in time with a song, and our code writes the key words and \
the stages' names. You are given the lesson the video must teach, its beats with the lines sung in each, \
each beat's steps as the script times them, what each drawing is meant to show, and frames from the whole film \
in order, each labelled with its second. In these frames every drawing holds still; in the film each one \
moves a little on the spot. The frames are moments, not the whole film: a step the script lists between two \
frames is in the film though you have no frame of it, so never report it as missing. The picture follows the \
song: showing what a line itself sings (three molts when the line sings of three) is never a fault.

List every problem a careful teacher would not let through:
- wrong_learning: a child would learn something false from the picture: it goes against what that beat \
teaches, or it shows what that beat or the lesson says must not be suggested or taught.
- hidden: a drawing covered by another drawing, a word or a name, or cut off by the frame's edge, so that it \
cannot be made out.
- unclear: what the lines sing about at that second is not on screen or cannot be recognised, or a stage's \
name stands under the wrong picture.
- stuck: the picture is the same, with nothing new on screen, in frames more than five seconds apart.

For each problem give the second of the first frame that shows it, its kind, and one sentence saying what \
is wrong and what a child would see or wrongly believe. List only what is plainly wrong in the frames. Do \
not comment on the drawing style, on taste, on sound, or on drawings holding still. An empty list means the \
film is fit to finish."""


def film_key(board: AnimationScript, drawings: dict[str, str]) -> str:
    """What a check is a check of: this script played with these drawings."""
    return hashlib.sha256("|".join([VERSION, json.dumps(board.model_dump(mode="json"), sort_keys=True),
                                    json.dumps(drawings, sort_keys=True)]).encode()).hexdigest()


def look_times(board: AnimationScript, wide: float) -> list[float]:
    """The seconds of the film the judge is shown, on a frame `wide` times as wide as it is tall."""
    scenes = scenes_of(board, board.beats, wide)
    tracks = [t for s in scenes for t in s.tracks]
    fades = sorted([(t.born, t.born + SETTLE) for t in tracks] + [(t.gone, t.gone + SETTLE) for t in tracks if t.out])

    def clear(at: float, until: float) -> float | None:
        """The first moment from `at` on when nothing is fading in or out, if it comes before `until`."""
        for start, end in fades:
            if start <= at < end:
                at = end
        return at if at < until else None

    arrived = sorted({t for track in tracks if (t := clear(track.born, track.gone)) is not None})
    apart = 0.0
    while True:                                                 # too many drawings to show each: spread them out
        times = [t for n, t in enumerate(arrived) if not n or t - arrived[n - 1] >= apart]
        if len(times) <= MAX_FRAMES:
            break
        apart += 0.25
    later = [clear(until, track.gone) for track in tracks for _, until, *_ in track.moves + track.zooms]
    later += [clear(at + SETTLE, until) for scene in scenes for at, until, *_ in scene.words]
    for t in sorted(t for t in later if t is not None):
        if len(times) < MAX_FRAMES and all(abs(t - chosen) >= APART for chosen in times):
            times.append(t)
    return sorted(times)


class FilmCheckService:
    """The check between the drawings and the loops: the film is put together once by code from the still
    drawings, which costs nothing, and a judge looks at frames from the whole song with the lesson beside
    them, for wrong learning, hidden drawings, unclear moments and stuck stretches. What it finds goes back
    to the script: the beats it names are written again, anything new is drawn, and the film is judged once
    more, up to REPAIRS times. A repair that leaves more problems is undone, so the script that is kept is
    the best one found; the film goes on with it, and what the judge still finds in it is stored and
    reported, not a stop. No loop is made until a check of the same script and drawings is done. A re-run
    with the same script and drawings reuses the stored findings."""

    def __init__(self, run_repository: RunRepository, song_repository: SongRepository,
                 asset_repository: AssetRepository, render: RenderService, storyboard: StoryboardService,
                 drawings: DrawingService, llm: LlmService, settings: Settings) -> None:
        self._runs = run_repository
        self._songs = song_repository
        self._assets = asset_repository
        self._render = render
        self._storyboard = storyboard
        self._drawings = drawings
        self._llm = llm
        self._settings = settings

    def run(self, run_id: str, accept: bool = False, force: bool = False) -> list[FilmProblem]:
        """accept: the findings have been read and the film may go on as it is."""
        if self._runs.runs.get(run_id) is None:
            raise LookupError(f"Run {run_id} not found")
        key = self._key(run_id)
        state = self._runs.stage(run_id, STAGE)
        if state is not None and not force and state.input_hash == key and state.output:
            logger.info("film check %s: reused, inputs unchanged", run_id[:8])
            if accept and state.status != StageStatus.DONE:
                state.status, state.output = StageStatus.DONE, {**state.output, "accepted": True}
                self._runs.stage_states.update(state)
                logger.warning("film check %s: %d problems accepted", run_id[:8], len(state.output["problems"]))
            return [FilmProblem(**p) for p in state.output["problems"]]
        state = state or self._runs.stage_states.add(StageState(run_id=run_id, stage=STAGE))
        state.status, state.attempts, state.error, state.output = StageStatus.RUNNING, state.attempts + 1, None, {}
        state.input_hash, state.started_at, state.finished_at = key, utcnow(), None
        state = self._runs.stage_states.update(state)
        try:
            problems, repairs = self._judge(run_id), 0
            while problems and not accept and repairs < REPAIRS:
                problems, repairs = self._mend(run_id, problems), repairs + 1
        except Exception as exc:
            fail_stage(self._runs, state, exc, logger, "film check")
            raise
        state.input_hash = self._key(run_id)                    # a repair has changed the script it is a check of
        state.status = StageStatus.DONE                         # with problems left, the best script found goes on
        state.error = "; ".join(f"{p.at:g} s {p.kind}: {p.what}" for p in problems) or None
        state.output = {"problems": [p.model_dump(mode="json") for p in problems], "repairs": repairs, "accepted": accept}
        state.finished_at = utcnow()
        self._runs.stage_states.update(state)
        logger.log(logging.WARNING if problems else logging.INFO, "film check %s: %s, %d repairs, %d problems left%s",
                   run_id[:8], state.status, repairs, len(problems),
                   "".join(f"\n  - {p.at:g} s {p.kind}: {p.what}" for p in problems))
        return problems

    def _key(self, run_id: str) -> str:
        drawn = self._runs.stage(run_id, StageName.VISUAL_IDENTITY)
        if drawn is None or drawn.status != StageStatus.DONE:
            raise ValueError(f"Run {run_id} has no finished drawings; run the drawings stage first")
        return film_key(script_of(self._runs, run_id), drawn.output["drawings"])

    def _judge(self, run_id: str) -> list[FilmProblem]:
        """The film as the stored script and drawings make it, looked at once."""
        board = script_of(self._runs, run_id)
        times, images = look_times(board, self._settings.assembly_width / self._settings.assembly_height), []
        for t, frame in zip(times, self._render.stills(run_id, times, HEIGHT)):
            data = io.BytesIO()
            frame.convert("RGB").save(data, "JPEG", quality=85)
            images.append(LabelledImage(label=f"the film at {t:.1f} s", data=data.getvalue(), media_type="image/jpeg"))
        review, call = self._llm.inspect(self._settings.film_check_model, SYSTEM, self._lesson(run_id, board), images,
                                         FilmReview, effort="medium")
        cost = add_llm_cost(self._assets, run_id, STAGE, call)
        problems = sorted(review.problems, key=lambda p: p.at)
        logger.info("film check %s: %d frames, %d problems, $%.2f%s", run_id[:8], len(times), len(problems), cost,
                    "".join(f"\n  - {p.at:g} s {p.kind}: {p.what}" for p in problems))
        return problems

    def _mend(self, run_id: str, problems: list[FilmProblem]) -> list[FilmProblem]:
        """One repair: the script's beats the problems fall in are written again, what is new is drawn, and
        the film is judged again. A repair that cannot be written or drawn, or that leaves the film with more
        problems than it had, is undone."""
        board = script_of(self._runs, run_id)
        beat = lambda at: next((b.beat for b in board.beats if b.start <= at < b.end), board.beats[-1].beat)
        try:
            self._storyboard.repair(run_id, [f"{beat(p.at)}: at {p.at:g} s the film shows this: {p.what}" for p in problems])
            self._drawings.run(run_id)
            again = self._judge(run_id)
        except (LlmError, ValueError) as exc:
            logger.warning("film check %s: no usable repair, keeping the script: %s", run_id[:8], exc)
            again = None
        if again is None or len(again) > len(problems):
            self._storyboard.restore(run_id, board)
            return problems
        return again

    def _lesson(self, run_id: str, board: AnimationScript) -> str:
        """What the film must teach and what its script does, beat by beat, and what each drawing is meant to show."""
        plan = LessonPlan(**self._runs.stage(run_id, StageName.LESSON).output)
        song = self._songs.songs.get(self._runs.runs.get(run_id).song_id)
        lines = {l.index: l.text for l in self._songs.analyses.get(song.analysis_id).lines}
        steps = {b.beat: "; ".join(_action(a) for a in b.actions if a.verb != Verb.LIPSYNC) for b in board.beats}
        beats = "\n".join(
            f"- {b.id}, {b.start:g} to {b.end:g} s. Sung: {' / '.join(lines[i] for i in b.lines) or 'music only'}."
            + (f" Teaches: {b.concept}" if b.concept else "") + (f" Must not suggest: {b.avoid}" if b.avoid else "")
            + f"\n  Steps: {steps[b.id]}" for b in plan.beats)
        drawings = "\n".join(f"- {d.id}{f', the stage named {d.name}' if d.name else ''}: {d.look}" for d in board.drawings)
        return (f"Audience: grades {plan.grade_band}.\nTopic: {plan.topic}. {plan.summary}\n"
                f"Key terms: {'; '.join(f'{t.term}: {t.definition}' for t in plan.key_terms)}\n"
                f"Must not teach: {' '.join(plan.must_not_teach) or 'nothing listed'}\n\nBeats:\n{beats}\n\n"
                f"Drawings:\n{drawings}")
