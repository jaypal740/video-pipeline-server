from pathlib import Path

import typer
import uvicorn

from src.core.container import Container
from src.core.logging import configure_logging
from src.core.database import create_tables
from src.schema.enums import StageName
from src.schema.lesson.lesson_schema import GradeBand
from src.schema.run.run_schema import Run
from src.schema.song.song_schema import SongStatus
from src.service.lesson.lesson_rules import band_for

app = typer.Typer(no_args_is_help=True)
db_app = typer.Typer(no_args_is_help=True, help="Database commands.")
song_app = typer.Typer(no_args_is_help=True, help="Songs: add one, analyse it (stage 1 intake), check it.")
run_app = typer.Typer(no_args_is_help=True, help="Video runs made from an analysed song.")
key_app = typer.Typer(no_args_is_help=True, help="API keys products call the server with.")
host_app = typer.Typer(no_args_is_help=True, help="The fixed host of each grade band: draw candidates, approve one.")
app.add_typer(db_app, name="db")
app.add_typer(song_app, name="song")
app.add_typer(run_app, name="run")
app.add_typer(key_app, name="key")
app.add_typer(host_app, name="host")

container = Container()


@app.callback()
def main() -> None:
    settings = container.settings()
    configure_logging(settings.log_level, settings.log_dir)


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
    monthly_cap: float | None = typer.Option(None, "--monthly-cap", help="Monthly spend cap in USD (saved, not enforced yet)."),
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
    lyrics: Path | None = typer.Option(None, "--lyrics", exists=True, dir_okay=False, help="Lyrics or narration text."),
    jamwise: Path | None = typer.Option(None, "--jamwise", exists=True, dir_okay=False,
                                        help="JamWise job export; its lyrics and timing are used instead of --lyrics."),
) -> None:
    """Register a song. Its audio is copied into the asset folder."""
    if (lyrics is None) == (jamwise is None):
        raise typer.BadParameter("give either --lyrics or --jamwise")
    song = container.song_service().add(name, audio, lyrics.read_text() if lyrics else None, jamwise)
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
    force: bool = typer.Option(False, "--force", help="Plan again even if nothing changed."),
) -> None:
    """Stage 2: the teaching beats: what each part of the song teaches, its key word and when it is sung."""
    plan = container.lesson_service().run(run_id, model=model, force=force)
    typer.echo(f"{plan.topic} ({plan.subject}), grades {plan.grade_band}")
    typer.echo(f"terms: {', '.join(t.term for t in plan.key_terms)}")
    for beat in plan.beats:
        typer.echo(f"  {beat.id}  {beat.start:6.1f}-{beat.end:6.1f}s  {beat.type:8s}  {beat.keyword or '-':16s}  "
                   f"{beat.concept[:80]}")


@run_app.command("storyboard")
def run_storyboard(
    run_id: str,
    model: str | None = typer.Option(None, "--model", help="Override the storyboard model."),
    force: bool = typer.Option(False, "--force", help="Board again even if nothing changed."),
) -> None:
    """Stage 3: the animation script: the drawings the video needs and what they do in each beat."""
    board = container.storyboard_service().run(run_id, model=model, force=force)
    typer.echo(f"grades {board.grade_band}: {board.style}")
    typer.echo(f"palette: {' '.join(board.palette)} | performer: {board.performer or '-'}")
    for d in board.drawings:
        typer.echo(f"  {d.id:20s} {d.kind:10s} {d.look[:60]}" + (f" | moves: {d.motion}" if d.motion else ""))
    for beat in board.beats:
        typer.echo(f"  {beat.beat}  {beat.start:6.1f}-{beat.end:6.1f}s  {beat.type:8s}  {beat.background}")
        for a in beat.actions:
            what = " ".join(str(v) for v in (a.drawing, f"to {a.to}" if a.to else None, a.pos, a.scale, a.text) if v)
            typer.echo(f"      {a.at:6.1f}{f'-{a.until:.1f}' if a.until is not None else '':7s} {a.verb:10s} {what}")
    typer.echo(f"review: {len(board.gaps)} gaps left" + "".join(f"\n  - {g}" for g in board.gaps))


@run_app.command("drawings")
def run_drawings(
    run_id: str,
    until: float | None = typer.Option(None, "--until", help="Only the drawings needed before this second of the song."),
    force: bool = typer.Option(False, "--force", help="Draw again even if nothing changed."),
) -> None:
    """Stage 4: every drawing of the animation script, drawn once in the band's style, cut out and checked."""
    made = container.drawing_service().run(run_id, until=until, force=force)
    for name, asset in made.items():
        typer.echo(f"  {name:26s} {asset.review:10s} try {asset.params.get('try', 1)}  {asset.params.get('cut') or asset.path}"
                   + (f"  ({asset.review_notes.splitlines()[0][:70]})" if asset.review_notes else ""))


@run_app.command("check")
def run_check(
    run_id: str,
    accept: bool = typer.Option(False, "--accept", help="The findings have been read; let the film go on as it is."),
    force: bool = typer.Option(False, "--force", help="Check again even if nothing changed."),
) -> None:
    """Between drawings and loops: the film put together from the still drawings and judged against the lesson."""
    problems = container.film_check_service().run(run_id, accept=accept, force=force)
    for p in problems:
        typer.echo(f"  {p.at:6.1f} s  {p.kind:14s} {p.what}")
    typer.echo(f"  {len(problems)} problems" + (", accepted" if problems and accept else ""))


@run_app.command("loops")
def run_loops(
    run_id: str,
    until: float | None = typer.Option(None, "--until", help="Only the loops needed before this second of the song."),
    force: bool = typer.Option(False, "--force", help="Animate again even if nothing changed."),
) -> None:
    """Stage 5: every drawing that has a motion, animated once into a short loop, cut out and checked."""
    made = container.loop_service().run(run_id, until=until, force=force)
    for name, asset in made.items():
        typer.echo(f"  {name:26s} {asset.review:10s} try {asset.params.get('try', 1)}  {asset.params.get('cut') or asset.path}"
                   + (f"  ({asset.review_notes.splitlines()[0][:70]})" if asset.review_notes else ""))


@run_app.command("render")
def run_render(
    run_id: str,
    until: float | None = typer.Option(None, "--until", help="Only the beats that start before this second of the song."),
    force: bool = typer.Option(False, "--force", help="Render again even if nothing changed."),
) -> None:
    """Stage 6: the script played by code from the drawings and loops, as one film with the song under it."""
    film = container.render_service().run(run_id, until=until, force=force)
    typer.echo(f"  {film.params['seconds']:.1f} s, {film.params['frames']} frames  {film.path}")


@run_app.command("sheets")
def run_sheets(
    run_id: str,
    force: bool = typer.Option(False, "--force", help="Draw again even if nothing changed."),
) -> None:
    """Stage 4a: one reference sheet per cast member and place, drawn in the band's style and checked."""
    sheets = container.sheet_service().run(run_id, force=force)
    for target, views in sheets.items():
        for view, asset in views.items():
            typer.echo(f"  {target:18s} {view:14s} {asset.review:12s} {asset.path}"
                       + (f"  ({asset.review_notes.splitlines()[0][:70]})" if asset.review_notes else ""))


@run_app.command("keyframes")
def run_keyframes(
    run_id: str,
    force: bool = typer.Option(False, "--force", help="Draw again even if nothing changed."),
) -> None:
    """Stage 4b: one picture per shot, the frozen opening moment, drawn from the sheets and checked."""
    made = container.keyframe_service().run(run_id, force=force)
    for shot_id, asset in made.items():
        typer.echo(f"  {shot_id}  {asset.review:12s} try {asset.params.get('try', 1)}  {asset.path}"
                   + (f"  ({asset.review_notes.splitlines()[0][:70]})" if asset.review_notes else ""))


@run_app.command("video")
def run_video(
    run_id: str,
    force: bool = typer.Option(False, "--force", help="Draw again even if nothing changed."),
) -> None:
    """Stage 5a: one clip per shot, animated from its keyframe and checked."""
    made = container.video_service().run(run_id, force=force)
    for shot_id, asset in made.items():
        typer.echo(f"  {shot_id}  {asset.review:12s} try {asset.params.get('try', 1)}  {asset.params.get('seconds', 0):.1f}s"
                   f"  {asset.params.get('score')}/10  {asset.path}"
                   + (f"  ({asset.review_notes.splitlines()[0][:70]})" if asset.review_notes else ""))


@run_app.command("lipsync")
def run_lipsync(
    run_id: str,
    force: bool = typer.Option(False, "--force", help="Draw again even if nothing changed."),
) -> None:
    """Stage 5b: the host singing each sung shot, lip-synced from its keyframe to the song."""
    made = container.lipsync_service().run(run_id, force=force)
    for shot_id, asset in made.items():
        typer.echo(f"  {shot_id}  {asset.review:12s} try {asset.params.get('try', 1)}  {asset.params.get('seconds', 0):.1f}s"
                   f"  {asset.params.get('score')}/10  {asset.path}"
                   + (f"  ({asset.review_notes.splitlines()[0][:70]})" if asset.review_notes else ""))


@run_app.command("assemble")
def run_assemble(
    run_id: str,
    force: bool = typer.Option(False, "--force", help="Assemble again even if nothing changed."),
) -> None:
    """Stage 6: every shot's clip on the song's timeline, with cuts and dissolves and the song laid on."""
    film = container.assembly_service().run(run_id, force=force)
    state = container.run_repository().stage(run_id, StageName.ASSEMBLY)
    for row in state.output["shots"]:
        typer.echo(f"  {row['shot']}  {row['source']:8s} x{row['speed']:.2f}  {row['frames']} frames")
    typer.echo(f"{state.status}: {state.output['seconds']:.1f}s  {film.path}"
               + (f"  ({'; '.join(state.output['problems'])})" if state.output["problems"] else ""))


@run_app.command("recheck-video")
def run_recheck_video(run_id: str) -> None:
    """Check the clips that did not pass again with the current rules; nothing is drawn."""
    for shot_id, asset in container.video_service().recheck(run_id).items():
        typer.echo(f"  {shot_id}  {asset.review:12s} try {asset.params.get('try', 1)}  {asset.params.get('score')}/10"
                   + (f"  ({asset.review_notes.splitlines()[0][:70]})" if asset.review_notes else ""))


@run_app.command("recheck")
def run_recheck(run_id: str) -> None:
    """Judge the keyframes that did not pass again with the current brief; nothing is drawn."""
    for shot_id, asset in container.keyframe_service().recheck(run_id).items():
        typer.echo(f"  {shot_id}  {asset.review:12s} try {asset.params.get('try', 1)}  {asset.params.get('score')}/10"
                   + (f"  ({asset.review_notes.splitlines()[0][:70]})" if asset.review_notes else ""))


@host_app.command("draw")
def host_draw(band: GradeBand, count: int = typer.Option(3, "--count", help="How many candidates to draw.")) -> None:
    """Draw candidate front views of a band's host from its description. Pick one with `host approve`."""
    for asset in container.sheet_service().host_candidates(band, count):
        typer.echo(f"  {asset.id}  candidate {asset.params['candidate']}  {'ok' if asset.review != 'failed' else 'FAILED'}"
                   f"  {asset.path}  ${asset.cost_usd:.3f}  {asset.params['seconds']}s")
        for line in (asset.review_notes or "").splitlines():
            typer.echo(f"      - {line}")


@host_app.command("approve")
def host_approve(band: GradeBand, asset_id: str) -> None:
    """Keep this candidate as the band's host, then draw its other views."""
    for asset in container.sheet_service().approve_host(band, asset_id):
        typer.echo(f"  {asset.params['view']:14s} {asset.review:12s} {asset.path}")


@host_app.command("recheck")
def host_recheck(band: GradeBand) -> None:
    """Run the image check again on the host's views waiting for review; draws nothing."""
    for asset in container.sheet_service().recheck_host(band):
        typer.echo(f"  {asset.params['view']:14s} {asset.review:12s}" + (f"  {asset.review_notes.splitlines()[0][:90]}" if asset.review_notes else ""))


@host_app.command("redraw")
def host_redraw(band: GradeBand, view: str) -> None:
    """Draw one view of the host again (front stays); the old tries are kept as failed."""
    asset = container.sheet_service().redraw_host_view(band, view)
    typer.echo(f"  {view:14s} {asset.review:12s} {asset.path}" + (f"  {asset.review_notes.splitlines()[0][:90]}" if asset.review_notes else ""))


@host_app.command("show")
def host_show(band: GradeBand) -> None:
    """The band's host images and their review state."""
    assets = container.asset_repository().assets.list(run_id=None)
    for asset in sorted((a for a in assets if a.target_id and a.target_id.startswith(f"host:{band}:")),
                        key=lambda a: a.created_at):
        typer.echo(f"  {asset.id}  {asset.params.get('view', ''):14s} {asset.review:12s} {asset.path}")


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
