from pathlib import Path

import typer

from src.core.container import Container
from src.core.database import create_tables
from src.schema.run.run_schema import Run

app = typer.Typer(no_args_is_help=True)
db_app = typer.Typer(no_args_is_help=True, help="Database commands.")
run_app = typer.Typer(no_args_is_help=True, help="Run commands.")
stage_app = typer.Typer(no_args_is_help=True, help="Run one stage of a run.")
intake_app = typer.Typer(no_args_is_help=True, help="Inspect the intake stage.")
app.add_typer(db_app, name="db")
app.add_typer(run_app, name="run")
app.add_typer(stage_app, name="stage")
app.add_typer(intake_app, name="intake")

container = Container()


@db_app.command("init")
def db_init() -> None:
    """Create every table."""
    engine = container.engine()
    create_tables(engine)
    typer.echo(f"Initialised {engine.url}")


@run_app.command("create")
def run_create(
    name: str = typer.Option(..., "--name", help="Working title."),
    audio: Path = typer.Option(..., "--audio", exists=True, dir_okay=False, help="Song audio file."),
    script: Path = typer.Option(..., "--script", exists=True, dir_okay=False, help="Lyrics or narration text."),
    audience: str | None = typer.Option(None, "--audience", help="Audience profile, e.g. teen."),
    brief: str | None = typer.Option(None, "--brief", help="What the video should teach."),
) -> None:
    """Register a new run with its inputs."""
    run = container.run_repository().runs.add(
        Run(name=name, audio_path=str(audio.resolve()), script_path=str(script.resolve()), audience=audience, brief=brief)
    )
    typer.echo(run.id)


@run_app.command("list")
def run_list() -> None:
    """Every run, oldest first."""
    for run in container.run_repository().runs.list():
        typer.echo(f"{run.id}  {run.status:8s}  {run.current_stage:16s}  {run.name}")


@run_app.command("show")
def run_show(run_id: str) -> None:
    """One run: inputs, stage states and spend so far."""
    run = container.run_repository().runs.get(run_id)
    if run is None:
        typer.echo(f"Run {run_id} not found")
        raise typer.Exit(code=1)
    typer.echo(run.model_dump_json(indent=2))
    for state in container.run_repository().stages(run_id):
        typer.echo(f"  {state.stage:16s} {state.status:12s} attempts={state.attempts}")
    typer.echo(f"  spent ${container.asset_repository().run_cost(run_id):.2f}")


@stage_app.command("intake")
def stage_intake(run_id: str, force: bool = typer.Option(False, "--force", help="Analyse again even if cached.")) -> None:
    """Stage 1: time every word of the lyrics and find the beats."""
    analysis = container.intake_service().run(run_id, force=force)
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


@intake_app.command("preview")
def intake_preview(run_id: str) -> None:
    """Write the check video: lines and words on screen as they are said, ticks on the beats."""
    run = container.run_repository().runs.get(run_id)
    if run is None:
        typer.echo(f"Run {run_id} not found")
        raise typer.Exit(code=1)
    analysis = container.intake_service().analysis_for(run_id)
    out_dir = Path(container.settings().asset_dir) / "runs" / run_id / "intake"
    typer.echo(container.preview_service().render(analysis, Path(run.audio_path), out_dir))


if __name__ == "__main__":
    app()
