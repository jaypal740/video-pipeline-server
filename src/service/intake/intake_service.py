import hashlib
import logging
from pathlib import Path

from src.repository.run.run_repository import RunRepository
from src.repository.song.song_repository import SongRepository
from src.schema.base_schema import utcnow
from src.schema.enums import StageName, StageStatus
from src.schema.run.stage_state_schema import StageState
from src.schema.song.song_analysis_schema import SongAnalysis
from src.service.intake.intake_alignment import REJECT_BELOW, lyric_lines, song_confidence, spoken_tokens, timed_words
from src.service.intake.intake_beats import beat_strength, bpm_of, onset_envelope
from src.service.intake.intake_lyrics import read_lyrics
from src.service.local.beat_service import SAMPLE_RATE as BEAT_RATE
from src.service.local.beat_service import BeatService
from src.service.local.ffmpeg_service import FfmpegService
from src.service.local.forced_align_service import SAMPLE_RATE as ALIGN_RATE
from src.service.local.forced_align_service import ForcedAlignService

logger = logging.getLogger(__name__)

# Bumped whenever the analysis changes, so every song is analysed again with the new code.
ANALYZER_VERSION = "intake.v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class IntakeService:
    """Stage 1: time every written word against the audio and find the beats. One analysis per
    audio + lyrics pair, reused by every run of the same song."""

    def __init__(self, run_repository: RunRepository, song_repository: SongRepository, ffmpeg: FfmpegService,
                 aligner: ForcedAlignService, beats: BeatService) -> None:
        self._runs = run_repository
        self._songs = song_repository
        self._ffmpeg = ffmpeg
        self._aligner = aligner
        self._beats = beats

    def run(self, run_id: str, force: bool = False) -> SongAnalysis:
        run = self._runs.runs.get(run_id)
        if run is None:
            raise LookupError(f"Run {run_id} not found")
        lyrics = read_lyrics(Path(run.script_path))
        if not lyrics:
            raise ValueError(f"{run.script_path} has no lyric lines")
        audio_sha = sha256_file(Path(run.audio_path))
        lyrics_sha = hashlib.sha256("\n".join(lyrics).encode()).hexdigest()
        key = hashlib.sha256(f"{ANALYZER_VERSION}|{audio_sha}|{lyrics_sha}".encode()).hexdigest()

        state = self._start(run_id, key)
        try:
            cached = self._songs.find_analysis(key)
            if cached is not None and not force:
                analysis, hit = cached, True
            else:
                analysis = self._analyse(Path(run.audio_path), lyrics, key, audio_sha, lyrics_sha)
                if cached is not None:
                    analysis.id, analysis.created_at = cached.id, cached.created_at
                    analysis = self._songs.analyses.update(analysis)
                else:
                    analysis = self._songs.analyses.add(analysis)
                hit = False
        except Exception as exc:
            self._finish(state, StageStatus.FAILED, error=str(exc) or type(exc).__name__)
            raise
        self._finish(state, StageStatus.DONE if analysis.accepted else StageStatus.FAILED,
                     output={"analysis_id": analysis.id, "cache_hit": hit}, error=analysis.reason)
        return analysis

    def analysis_for(self, run_id: str) -> SongAnalysis:
        state = self._runs.stage(run_id, StageName.INTAKE)
        if state is None or "analysis_id" not in state.output:
            raise LookupError(f"Run {run_id} has no intake analysis; run the intake stage first")
        return self._songs.analyses.get(state.output["analysis_id"])

    def _analyse(self, audio: Path, lyrics: list[str], key: str, audio_sha: str, lyrics_sha: str) -> SongAnalysis:
        speech = self._ffmpeg.decode(audio, ALIGN_RATE)
        tokens, owners = spoken_tokens(lyrics)
        words = timed_words(lyrics, owners, self._aligner.align(speech, tokens))
        lines = lyric_lines(lyrics, words)
        confidence = song_confidence(lines)
        accepted = confidence >= REJECT_BELOW
        beats, downbeats = self._beats.track(self._ffmpeg.decode(audio, BEAT_RATE))
        bpm = bpm_of(beats)
        logger.info("intake: %d lines, confidence %.2f, %d unclear, %s bpm", len(lines), confidence,
                    sum(l.unclear for l in lines), bpm)
        return SongAnalysis(
            cache_key=key, audio_sha256=audio_sha, lyrics_sha256=lyrics_sha,
            duration_s=round(len(speech) / ALIGN_RATE, 3), accepted=accepted,
            reason=None if accepted else (f"the audio matches these lyrics at {confidence:.2f}, below "
                                          f"{REJECT_BELOW:.2f}: check that this is the right audio for this script"),
            words=words, lines=lines, bpm=bpm, beats=beats, downbeats=downbeats,
            beat_strength=beat_strength(onset_envelope(speech, ALIGN_RATE), bpm),
            tool_versions={"analyzer": ANALYZER_VERSION, "aligner": self._aligner.version,
                           "beats": self._beats.version, "confidence": confidence})

    def _start(self, run_id: str, key: str) -> StageState:
        state = self._runs.stage(run_id, StageName.INTAKE)
        if state is None:
            state = self._runs.stage_states.add(StageState(run_id=run_id, stage=StageName.INTAKE))
        state.status, state.attempts, state.input_hash = StageStatus.RUNNING, state.attempts + 1, key
        state.started_at, state.finished_at, state.error = utcnow(), None, None
        return self._runs.stage_states.update(state)

    def _finish(self, state: StageState, status: StageStatus, output: dict | None = None, error: str | None = None) -> None:
        state.status, state.finished_at, state.error = status, utcnow(), error
        if output is not None:
            state.output = output
        self._runs.stage_states.update(state)
