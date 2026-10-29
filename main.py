from pathlib import Path

import typer
import uvicorn

from src.core.container import Container
from src.core.database import create_tables
from src.schema.lesson.lesson_schema import VideoType
from src.schema.run.run_schema import Run
from src.schema.song.song_schema import SongStatus
from src.service.lesson.lesson_rules import band_for

app = typer.Typer(no_args_is_help=True)
db_app = typer.Typer(no_args_is_help=True, help="Database commands.")
song_app = typer.Typer(no_args_is_help=True, help="Songs: add one, analyse it (stage 1 intake), check it.")
run_app = typer.Typer(no_args_is_help=True, help="Video runs made from an analysed song.")
key_app = typer.Typer(no_args_is_help=True, help="API keys products call the server with.")
app.add_typer(db_app, name="db")
app.add_typer(song_app, name="song")
app.add_typer(run_app, name="run")
app.add_typer(key_app, name="key")

container = Container()


@db_app.command("init")
def db_init() -> None:
    """Create every table."""
    engine = container.engine()
    create_tables(engine)
    typer.echo(f"Initialised {engine.url}")


@app.command("serve")
def serve(host: str = typer.Option("127.0.0.1", "--host"), port: int = typer.Option(8000, "--port")) -> None:
    """Start the API server."""
    uvicorn.run("src.api.app:app", host=host, port=port)


@key_app.command("create")
def key_create(
    name: str = typer.Argument(..., help="The product the key is for, e.g. jamwise."),
    monthly_cap: float | None = typer.Option(None, "--monthly-cap", help="Monthly spend cap in USD."),
) -> None:
    """Issue a key. It is printed once and never stored; copy it now."""
    key, raw = container.auth_service().create(name, monthly_cap)
    typer.echo(f"{key.prefix}  {key.name}\n{raw}")


@key_app.command("list")
def key_list() -> None:
    """Every key, by its prefix."""
    for key in container.auth_service().list():
        state = "revoked" if key.revoked_at else "active"
        typer.echo(f"{key.prefix}  {state:7s}  {key.name}  last used {key.last_used_at or 'never'}")


@key_app.command("revoke")
def key_revoke(prefix: str) -> None:
    """Revoke a key by its prefix; requests with it are refused from now on."""
    key = container.auth_service().revoke(prefix)
    typer.echo(f"{key.prefix} revoked")


@song_app.command("add")
def song_add(
    name: str = typer.Option(..., "--name", help="Unique song name, e.g. antibiotic-resistance."),
    audio: Path = typer.Option(..., "--audio", exists=True, dir_okay=False, help="Song audio file."),
    lyrics: Path = typer.Option(..., "--lyrics", exists=True, dir_okay=False, help="Lyrics or narration text."),
) -> None:
    """Register a song. Its audio is copied into the asset folder."""
    song = container.song_service().add(name, audio, lyrics.read_text())
    typer.echo(song.id)


@song_app.command("list")
def song_list() -> None:
    """Every song, oldest first."""
    for song in container.song_service().list():
        typer.echo(f"{song.id}  {song.status:9s}  {song.name}")


@song_app.command("analyse")
def song_analyse(song: str, force: bool = typer.Option(False, "--force", help="Analyse again even if cached.")) -> None:
    """Stage 1 intake: time every word of the lyrics and find the beats. SONG is its id or name."""
    analysis = container.intake_service().run(song, force=force)
    unclear = [l for l in analysis.lines if l.unclear]
    typer.echo(f"{'accepted' if analysis.accepted else 'REJECTED'}: {len(analysis.lines)} lines, {len(analysis.words)} words, "
               f"confidence {analysis.tool_versions.get('confidence', 0):.2f}, {len(unclear)} unclear, "
               f"{analysis.bpm or 0:.0f} BPM, beat strength {analysis.beat_strength:.2f}")
    if analysis.reason:
        typer.echo(analysis.reason)
    for line in analysis.lines:
        flag = "  UNCLEAR" if line.unclear else ""
        typer.echo(f"  {line.index:3d}  {line.start:7.2f}  {line.end:7.2f}  {line.confidence:.2f}  {line.text[:60]}{flag}")
    if not analysis.accepted:
        raise typer.Exit(code=1)


@song_app.command("preview")
def song_preview(song: str) -> None:
    """Write the check video: lines and words on screen as they are said, ticks on the beats."""
    record = container.song_service().get(song)
    analysis = container.intake_service().analysis_for(record.id)
    out_dir = Path(container.settings().asset_dir) / "songs" / record.id / "intake"
    typer.echo(container.preview_service().render(analysis, Path(record.audio_path), out_dir))


@run_app.command("create")
def run_create(
    song: str = typer.Option(..., "--song", help="Song id or name; it must be analysed first."),
    name: str | None = typer.Option(None, "--name", help="Working title; defaults to the song name."),
    grade: int | None = typer.Option(None, "--grade", min=0, max=8, help="Lowest grade it's for; 0 is kindergarten."),
    brief: str | None = typer.Option(None, "--brief", help="What the video should teach."),
) -> None:
    """Start a video run from an analysed song."""
    record = container.song_service().get(song)
    if record.status != SongStatus.READY:
        typer.echo(f"Song {record.name!r} is {record.status}; run `song analyse {record.name}` first")
        raise typer.Exit(code=1)
    run = container.run_repository().runs.add(
        Run(song_id=record.id, name=name or record.name, audience=band_for(grade) if grade is not None else None,
            brief=brief))
    typer.echo(run.id)


@run_app.command("lesson")
def run_lesson(
    run_id: str,
    model: str | None = typer.Option(None, "--model", help="Override the lesson model."),
    video_type: VideoType | None = typer.Option(None, "--type", help="Ask for this video type."),
    force: bool = typer.Option(False, "--force", help="Plan again even if nothing changed."),
) -> None:
    """Stage 2: what each line teaches and shows, and the video type for the run's grade band."""
    plan = container.lesson_service().run(run_id, model=model, video_type=video_type, force=force)
    typer.echo(f"{plan.topic} ({plan.subject}), grades {plan.grade_band}: {plan.video_type}. {plan.type_reason}")
    typer.echo(f"terms: {', '.join(t.term for t in plan.key_terms)}")
    for line in plan.lines:
        typer.echo(f"  {line.index:3d}  {line.tag:14s}  {line.show[:90]}")


@run_app.command("storyboard")
def run_storyboard(
    run_id: str,
    model: str | None = typer.Option(None, "--model", help="Override the storyboard model."),
    force: bool = typer.Option(False, "--force", help="Board again even if nothing changed."),
) -> None:
    """Stage 3: the shot list, cut on the beat, that the image and video models draw."""
    board = container.storyboard_service().run(run_id, model=model, force=force)
    typer.echo(f"{board.video_type}, grades {board.grade_band}: {board.shape}")
    typer.echo(f"places: {', '.join(p.id for p in board.places)} | cast: {', '.join(c.id for c in board.cast)}")
    for shot in board.shots:
        cam = " ".join(str(v) for v in (shot.camera.move, shot.camera.direction, shot.camera.speed) if v)
        typer.echo(f"  {shot.id} {shot.start:6.1f}-{shot.end:5.1f}  {shot.kind:10s} {shot.place:16s} {shot.size:13s} "
                   f"{cam:18s} {shot.action[:60]}")
    for flag in board.flags:
        typer.echo(f"  note: {flag}")


@run_app.command("list")
def run_list() -> None:
    """Every run, oldest first."""
    for run in container.run_repository().runs.list():
        typer.echo(f"{run.id}  {run.status:8s}  {run.current_stage:16s}  {run.name}")


@run_app.command("show")
def run_show(run_id: str) -> None:
    """One run: its song, stage states and spend so far."""
    run = container.run_repository().runs.get(run_id)
    if run is None:
        typer.echo(f"Run {run_id} not found")
        raise typer.Exit(code=1)
    typer.echo(run.model_dump_json(indent=2))
    for state in container.run_repository().stages(run_id):
        typer.echo(f"  {state.stage:16s} {state.status:12s} attempts={state.attempts}")
    typer.echo(f"  spent ${container.asset_repository().run_cost(run_id):.2f}")


if __name__ == "__main__":
    app()
