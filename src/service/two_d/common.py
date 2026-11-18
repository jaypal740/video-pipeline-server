import tomllib
from pathlib import Path

from src.repository.run.run_repository import RunRepository
from src.schema.enums import FilmType, StageName, StageStatus
from src.schema.lesson.lesson_schema import GradeBand
from src.schema.two_d.script_schema import AnimationScript
from src.service.stage.stage_common import require_film_type

TWO_D = frozenset({FilmType.TWO_D})
# The flat cartoon look of 2d films, per grade band. The band files hold the realistic films' painted look.
STYLES = Path(__file__).resolve().parents[2] / "templates" / "two_d" / "style.toml"


def script_of(runs: RunRepository, run_id: str) -> AnimationScript:
    require_film_type(runs, run_id, TWO_D)
    state = runs.stage(run_id, StageName.STORYBOARD)
    if state is None or state.status != StageStatus.DONE:
        raise ValueError(f"Run {run_id} has no animation script; run `run 2d storyboard` first")
    return AnimationScript(**state.output)


def style_of(band: GradeBand) -> str:
    """The look every drawing of a 2d film is made in."""
    style = tomllib.loads(STYLES.read_text())[str(band)]
    return f"{style['style_name']} with {style['style']}"
