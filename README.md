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
poetry run python main.py run create --name "Ocean" --audio song.mp3 --script lyrics.txt
poetry run python main.py run list
poetry run python main.py run show <run_id>
```

## Layout

| Folder | Purpose |
|---|---|
| `src/core` | settings, DI container, database engine, unit of work |
| `src/model` | SQLAlchemy tables, one file per table |
| `src/schema` | Pydantic models, mirrors `model/` |
| `src/repository` | the only layer that touches the database |
| `src/service` | stage logic and provider wrappers (added stage by stage) |
| `tests` | in-memory SQLite, container overrides instead of patching |

Stages: intake, lesson, beats, storyboard, visual identity, keyframes, video, lipsync, assembly, qc.
