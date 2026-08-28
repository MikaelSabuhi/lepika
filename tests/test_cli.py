from __future__ import annotations

from typer.testing import CliRunner

from lepika.cli import app

runner = CliRunner()


def test_version_flag_prints_name_and_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == "lepika 0.1.0"
