from typer.testing import CliRunner

from main import app


def test_run_2d_lists_the_2d_stages():
    result = CliRunner().invoke(app, ["run", "2d", "--help"])
    assert result.exit_code == 0
    for command in ("lesson", "storyboard", "drawings", "check", "loops", "render"):
        assert command in result.output
