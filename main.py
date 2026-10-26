from pathlib import Path

import typer

from src.core.container import Container
from src.core.database import create_tables
from src.schema.run.run_schema import Run

app = typer.Typer(no_args_is_help=True)
db_app = typer.Typer(no_args_is_help=True, help="Database commands.")
run_app = typer.Typer(no_args_is_help=True, help="Run commands.")
app.add_typer(db_app, name="db")
app.add_typer(run_app, name="run")

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


if __name__ == "__main__":
    app()
