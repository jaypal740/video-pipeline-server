# video-pipeline

Turns a song and its lyrics into an educational music video. Rebuilt stage by stage from the JamWise
pipeline; each stage is brought to production quality before the next one is built.

## Setup

```bash
poetry install                     # Python 3.12, venv in .venv/
cp .env.example .env               # add FAL_KEY and OPENAI_API_KEY
poetry run python main.py db init  # creates data/pipeline.db
poetry run pytest -q
```

## Usage

```bash
poetry run python main.py song add --name antibiotic-resistance --audio audio.mp3 --lyrics lyrics.txt
poetry run python main.py song analyse antibiotic-resistance     # stage 1: word timings, beats
poetry run python main.py song preview antibiotic-resistance     # check video: words highlighted as said, beat ticks
poetry run python main.py song list
poetry run python main.py run create --song antibiotic-resistance --audience teen
poetry run python main.py run list
poetry run python main.py run show <run_id>
```

A song is analysed once; every video run made from it reuses the analysis.

## Stage 1: intake

Times every written word of a song against its audio and finds the beats. One analysis per audio +
lyrics pair, cached by their hashes.

- Word timing: forced alignment (WhisperX, wav2vec2). Numbers are spoken as words first (1858 ->
  eighteen fifty eight), since the aligner only knows letters.
- A line below 0.5 confidence is marked unclear; a song averaging below 0.5 is rejected as the wrong audio.
- Beats and downbeats: beat_this. `beat_strength` near 0 means the music has no steady pulse
  (narration beds), so later stages cut on the voice instead of the beat.

## Layout

| Folder | Purpose |
|---|---|
| `src/core` | settings, DI container, database engine, unit of work |
| `src/model` | SQLAlchemy tables, one file per table |
| `src/schema` | Pydantic models, mirrors `model/` |
| `src/repository` | the only layer that touches the database |
| `src/service/<stage>` | one folder per stage: `<stage>_service.py` plus `<stage>_*.py` helpers |
| `src/service/local` | wrappers for local tools (ffmpeg, forced alignment, beat_this) |
| `tests` | `test_<module>.py`; in-memory SQLite, container overrides instead of patching |
| `data/` | database and generated media (git-ignored) |
| `scratch/` | temporary test files, deleted once a test passes (git-ignored) |

Stages: intake, lesson, beats, storyboard, visual identity, keyframes, video, lipsync, assembly, qc.
