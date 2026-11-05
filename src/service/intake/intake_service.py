import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

from src.repository.song.song_repository import SongRepository
from src.schema.song.song_analysis_schema import SongAnalysis
from src.schema.song.song_schema import Song, SongStatus
from src.service.intake.intake_alignment import REJECT_BELOW, lyric_lines, song_confidence, spoken_tokens, timed_words
from src.service.intake.intake_beats import beat_strength, bpm_of, onset_envelope
from src.service.intake.intake_jamwise import JAMWISE_FILE, jamwise_lines, read_export
from src.service.intake.intake_lyrics import parse_lyrics
from src.service.local.beat_service import SAMPLE_RATE as BEAT_RATE
from src.service.local.beat_service import BeatService
from src.service.local.ffmpeg_service import FfmpegService
from src.service.local.forced_align_service import SAMPLE_RATE as ALIGN_RATE
from src.service.local.forced_align_service import ForcedAlignService

logger = logging.getLogger(__name__)

# Bumped whenever the analysis changes, so every song is analysed again with the new code.
ANALYZER_VERSION = "intake.v2"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _longest(a: datetime | None, b: datetime | None) -> datetime | None:
    """The later of two expiry times; None (never expires) beats any time."""
    return None if a is None or b is None else max(a, b)


class IntakeService:
    """Stage 1: time every written word of a song against its audio and find the beats. One analysis
    per audio + lyrics pair: a second song with the same audio and words reuses it."""

    def __init__(self, song_repository: SongRepository, ffmpeg: FfmpegService, aligner: ForcedAlignService,
                 beats: BeatService) -> None:
        self._songs = song_repository
        self._ffmpeg = ffmpeg
        self._aligner = aligner
        self._beats = beats

    def analyse(self, audio: Path, lyrics_text: str, force: bool = False,
                expires_at: datetime | None = None, timing: list[dict] | None = None) -> SongAnalysis:
        """Analyse any audio + lyrics pair, a library song's or a job's. expires_at None keeps the
        analysis forever; a shared analysis keeps the longest retention any of its users asked for.
        timing: JamWise's lineTiming, used instead of aligning the words ourselves."""
        parsed = parse_lyrics(lyrics_text)
        if not parsed:
            raise ValueError("The lyrics have no lines to align")
        lyrics, sections = [line for _, line in parsed], [section for section, _ in parsed]
        audio_sha = sha256_file(audio)
        lyrics_sha = hashlib.sha256("\n".join(f"{s or ''}|{l}" for s, l in parsed).encode()).hexdigest()
        key = f"{ANALYZER_VERSION}|{audio_sha}|{lyrics_sha}"
        if timing is not None:
            key += "|" + hashlib.sha256(json.dumps(timing, sort_keys=True).encode()).hexdigest()
        key = hashlib.sha256(key.encode()).hexdigest()

        cached = self._songs.find_analysis(key)
        if cached is not None and not force:
            keep = _longest(cached.expires_at, expires_at)
            if keep != cached.expires_at:
                cached.expires_at = keep
                cached = self._songs.analyses.update(cached)
            return cached
        analysis = self._analyse(audio, lyrics, sections, key, audio_sha, lyrics_sha, timing)
        if cached is None:
            analysis.expires_at = expires_at
            return self._songs.analyses.add(analysis)
        analysis.id, analysis.created_at = cached.id, cached.created_at
        analysis.expires_at = _longest(cached.expires_at, expires_at)
        return self._songs.analyses.update(analysis)

    def run(self, song_ref: str, force: bool = False) -> SongAnalysis:
        """Analyse a library song and record the outcome on it."""
        song = self._song(song_ref)
        song.status, song.reason = SongStatus.ANALYSING, None
        song = self._songs.songs.update(song)
        try:
            export = Path(song.audio_path).parent / JAMWISE_FILE
            timing = read_export(export)["lineTiming"] if export.is_file() else None
            analysis = self.analyse(Path(song.audio_path), song.lyrics, force=force, timing=timing)
        except Exception as exc:
            song.status, song.reason = SongStatus.FAILED, str(exc) or type(exc).__name__
            self._songs.songs.update(song)
            raise
        song.status = SongStatus.READY if analysis.accepted else SongStatus.REJECTED
        song.analysis_id, song.reason = analysis.id, analysis.reason
        self._songs.songs.update(song)
        return analysis

    def analysis_for(self, song_ref: str) -> SongAnalysis:
        song = self._song(song_ref)
        if song.analysis_id is None:
            raise LookupError(f"Song {song.name!r} has not been analysed; run intake on it first")
        return self._songs.analyses.get(song.analysis_id)

    def _song(self, ref: str) -> Song:
        song = self._songs.find_song(ref)
        if song is None:
            raise LookupError(f"Song {ref} not found")
        return song

    def _analyse(self, audio: Path, lyrics: list[str], sections: list[str | None], key: str, audio_sha: str,
                 lyrics_sha: str, timing: list[dict] | None) -> SongAnalysis:
        speech = self._ffmpeg.decode(audio, ALIGN_RATE)
        if timing is None:
            tokens, owners = spoken_tokens(lyrics)
            words = timed_words(lyrics, owners, self._aligner.align(speech, tokens))
            lines = lyric_lines(lyrics, words)
        else:
            words, lines = jamwise_lines(timing, lyrics, len(speech) / ALIGN_RATE)
        for line, section in zip(lines, sections):
            line.section = section
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
            tool_versions={"analyzer": ANALYZER_VERSION, "aligner": "jamwise" if timing else self._aligner.version,
                           "beats": self._beats.version, "confidence": confidence})
