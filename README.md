# video-pipeline

Turns a song and its lyrics into an educational music video. Rebuilt stage by stage from the JamWise
pipeline; each stage is brought to production quality before the next one is built.

## Prerequisites

- **Python 3.12** (`pyproject.toml` requires `>=3.12,<3.13`)
- **Poetry**
- **ffmpeg and ffprobe** on your `PATH`; every stage from intake to assembly calls them
- API keys in `.env`:
  - `FAL_KEY`: every image, video and lip-sync model runs through fal
  - `CLAUDE_API_KEY`: the default lesson, storyboard and image-check models are Claude models
  - `OPENAI_API_KEY`: needed only if you set an LLM stage to a model whose name doesn't start with `claude-`

## Setup

```bash
poetry install                     # venv in .venv/
cp .env.example .env               # add FAL_KEY and CLAUDE_API_KEY
poetry run python main.py db init  # creates data/pipeline.db
poetry run pytest -q
```

## Making a video

```bash
# 1. Add a song and analyse it (stage 1). A song is analysed once; every run made from it reuses the analysis.
poetry run python main.py song add --name antibiotic-resistance --audio audio.mp3 --lyrics lyrics.txt
poetry run python main.py song analyse antibiotic-resistance
poetry run python main.py song preview antibiotic-resistance     # check video: words highlighted as said, beat ticks

# 2. Once per grade band (k-2, 3-5, 6-8): draw host candidates, then approve one that passed its check.
poetry run python main.py host draw 3-5 --count 3
poetry run python main.py host approve 3-5 <asset_id>

# 3. Create a run. --grade is the lowest grade it's for (0 = kindergarten, up to 8); it sets the grade band.
poetry run python main.py run create --song antibiotic-resistance --grade 4

# 4. Run the stages in order.
poetry run python main.py run lesson     <run_id>   # stage 2
poetry run python main.py run storyboard <run_id>   # stage 3
poetry run python main.py run sheets     <run_id>   # stage 4a (needs an approved host for the run's band)
poetry run python main.py run keyframes  <run_id>   # stage 4b
poetry run python main.py run video      <run_id>   # stage 5a
poetry run python main.py run lipsync    <run_id>   # stage 5b (needed when the board has sung shots)
poetry run python main.py run assemble   <run_id>   # stage 6
```

Every stage command takes `--force` to run again even when its inputs haven't changed. `run lesson` also
takes `--model` and `--type` (`concept`, `process`, `journey`, `personalised_hero`), and `run storyboard`
takes `--model`.

### Other commands

| Command | What it does |
|---|---|
| `song list` | Every song, oldest first |
| `run list` | Every run, oldest first |
| `run show <run_id>` | One run: its stage states and spend so far |
| `run recheck <run_id>` | Judges again the keyframes that didn't pass, with the current brief; draws nothing |
| `run recheck-video <run_id>` | Checks again the clips that didn't pass, with the current rules; draws nothing |
| `host show <band>` | The band's host images and their review state |
| `host recheck <band>` | Runs the image check again on the host's views waiting for review; draws nothing |
| `host redraw <band> <view>` | Draws one view of the host again (the front stays); old tries are kept as failed |
| `key create <name> [--monthly-cap N]` | Issues an API key; it is printed once and never stored. The monthly cap is saved but **not enforced yet** |
| `key list` / `key revoke <prefix>` | Lists keys / revokes one |
| `serve [--host] [--port]` | Starts the API server (default `127.0.0.1:8000`) |
| `db init` | Creates every table |

## Stages

Each stage records its status, attempts and error in the database (`run show`). A stage whose inputs are
unchanged since its last successful run is reused instead of paid for again.

**Stage 1: intake** (`song analyse`). Times every written word of a song against its audio and finds the
beats. One analysis per audio + lyrics pair, cached by their hashes.

- Word timing: forced alignment (WhisperX, wav2vec2). Numbers are spoken as words first (1858 ->
  eighteen fifty eight), since the aligner only knows letters.
- A line below 0.5 confidence is marked unclear; a song averaging below 0.5 is rejected as the wrong audio.
- Beats and downbeats: beat_this. `beat_strength` near 0 means the music has no steady pulse
  (narration beds), so later stages cut on the voice instead of the beat.

**Stage 2: lesson** (`run lesson`). From the timed lyrics, what each line teaches and shows, and which video
type fits the grade band. The model drafts; code checks the draft against the band and type rules and
re-asks until it passes. Needs the run to have a grade (`run create --grade`).

**Stage 3: storyboard** (`run storyboard`). From the lesson plan and the song's timing, the shot list the
image and video models draw. Code cuts the shots on the beat; the model writes the board; code fixes what it
can, checks the rest and re-asks only the shots that fail.

**Stage 4a: sheets** (`run sheets`). One reference sheet per cast member and place. Each band's host is drawn
once and approved by a person (`host draw`, `host approve`); every other sheet in that band is drawn with the
host's front view as a style reference, then judged (see below).

**Stage 4b: keyframes** (`run keyframes`). One picture per shot, the opening moment, drawn from the sheets and
checked. Shots in the same place form a chain: each later opening is drawn from the picture the shot before
ended on, so the cut joins up; a shot where the subject moves also gets an ending picture.

**Stage 5a: video** (`run video`). One clip per shot. A shot where something moves is animated by the video
model between its start and end keyframes, then checked by code (does it move, does it freeze) and by the
judge on three frames; its audio is dropped. A shot where only the camera moves is an ffmpeg move over its
keyframe. Sung shots are skipped here (`PIPELINE_VIDEO_SUNG_MODE=skip`, the default) and made by lip-sync.

**Stage 5b: lipsync** (`run lipsync`). One clip per sung shot of the host singing the line, made by the
lip-sync model from the shot's keyframe and the song. Checked by code (length, black frames), then by the
judge on the middle frame against the keyframe. Stored silent.

**Stage 6: assembly** (`run assemble`). Every shot's clip on the song's timeline, with the board's cuts and
dissolves and the song laid on, as one film. Model clips are retimed to fill their slot; lip-sync clips are
never retimed, so the lips stay on the words. The film is stored at `runs/<run_id>/film/` in the asset store
(`data/assets/` with local storage).

### How images and clips are judged

In the sheets, keyframes, video and lipsync stages, a vision model (the judge) scores each picture or clip
from 0 to 10. No person decides, except when choosing the host:

- A try passes when its score is at or above `PIPELINE_IMAGE_PASS_SCORE` (default 6) and it breaks no fatal
  rule (such as text laid on the picture, or a person who isn't in the cast).
- Otherwise it is drawn again, up to `PIPELINE_IMAGE_MAX_TRIES` draws (default 2). In the keyframes and video
  stages the retry carries a correction written from the judge's findings.
- If no try passes, the best try at or above `PIPELINE_IMAGE_FLOOR_SCORE` (default 5) is used; below that it
  fails.
- The stop-loss (keyframes and video): if more than `PIPELINE_IMAGE_STOP_LOSS` (default 0.5) of the stage's
  first 8 finished shots failed every draw, the stage stops drawing.

### Spend cap

Before drawing, the keyframe, video and lipsync stages work out the most the stage could cost (every try
used) and refuse to start if the run would pass `PIPELINE_SPEND_CAP_USD` (default $25).

## Configuration

Every setting is an environment variable read from `.env`; `.env.example` lists them all with their
defaults, and `src/core/configs.py` explains each one. The default models:

| Setting | Default |
|---|---|
| `PIPELINE_LESSON_MODEL`, `PIPELINE_STORYBOARD_MODEL` | `claude-opus-5-5` |
| `PIPELINE_IMAGE_CHECK_MODEL` (the judge) | `claude-haiku-5-5` |
| `PIPELINE_IMAGE_SECOND_OPINION_MODEL` | `claude-sonnet-5-5` |
| `PIPELINE_SHEET_MODEL`, `PIPELINE_KEYFRAME_MODEL` | `nano-banana-2.1` |
| `PIPELINE_HOST_MODEL`, `PIPELINE_IMAGE_FALLBACK_MODEL` | `gpt-image-2` |
| `PIPELINE_VIDEO_MODEL` / `PIPELINE_VIDEO_FALLBACK_MODEL` | `minimax-h3-max-turbo` / `kling-o3-standard` |
| `PIPELINE_LIPSYNC_MODEL` | `minimax-h3-max-lipsync` |

Storage is local files by default (`PIPELINE_STORAGE=local`, under `PIPELINE_ASSET_DIR`); set
`PIPELINE_STORAGE=r2` and the four `R2_` values to use Cloudflare R2.

## API

`main.py serve` starts a FastAPI server. Requests other than `/health` need an API key from `key create`, sent
as `Authorization: Bearer <key>`.

| Route | What it returns |
|---|---|
| `GET /health` | Liveness check; needs no key |
| `GET /v1/songs` | Library songs, oldest first |
| `GET /v1/songs/{song_ref}` | One song by its id or name |

Runs and stages can't be started through the API yet; they run from the CLI.

## Layout

| Folder | Purpose |
|---|---|
| `main.py` | the CLI (Typer) |
| `src/api` | the FastAPI app, routers and API-key dependency |
| `src/core` | settings, DI container, database engine, unit of work, logging |
| `src/model` | SQLAlchemy tables, one file per table |
| `src/schema` | Pydantic models, mirrors `model/` |
| `src/repository` | the only layer that touches the database |
| `src/service/<stage>` | one folder per stage: `<stage>_service.py` plus `<stage>_*.py` helpers |
| `src/service/stage` | steps the stages share (failing a stage, keeping the best try, stop-loss, costs) |
| `src/service/local` | wrappers for local tools (ffmpeg, forced alignment, beat_this) |
| `src/service/{llm,image,storage,auth,song}` | model providers, image pricing and judging, file storage, API keys, songs |
| `src/templates` | per-band, per-format and per-host settings (TOML) |
| `tests` | `test_<module>.py`; in-memory SQLite, container overrides instead of patching |
| `data/` | database, logs and generated media (git-ignored) |
| `scratch/` | temporary test files, deleted once a test passes (git-ignored) |

Built stages: intake, lesson, storyboard, sheets, keyframes, video, lipsync, assembly. `beats` and `qc` are
named in `StageName` (`src/schema/enums.py`) but have no service yet.
