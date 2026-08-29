from __future__ import annotations

import io
import sys

import pytest
from typer.testing import CliRunner

from lepika import cli
from lepika.cli import app

runner = CliRunner()


def test_version_flag_prints_name_and_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == "lepika 0.1.0"


def _cp1252_stream() -> io.TextIOWrapper:
    # What Python hands a Windows process whose stdout is a pipe: the ANSI code page.
    return io.TextIOWrapper(io.BytesIO(), encoding="cp1252", write_through=True)


def test_utf8_stdio_reconfigures_a_non_utf8_wrapper() -> None:
    stream = _cp1252_stream()
    cli._utf8_stdio([stream])
    stream.write("✓")
    stream.flush()
    assert stream.buffer.getvalue() == "✓".encode()


def test_utf8_stdio_leaves_a_utf8_wrapper_alone() -> None:
    stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", write_through=True)
    cli._utf8_stdio([stream])
    assert stream.encoding == "utf-8"


def test_utf8_stdio_skips_streams_that_cannot_be_reconfigured() -> None:
    cli._utf8_stdio([io.StringIO()])  # no TextIOWrapper, no error


def test_run_prints_a_check_mark_to_a_cp1252_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = _cp1252_stream()
    monkeypatch.setattr(sys, "stdout", stream)
    # Rich's Console reads sys.stdout at write time, which is what makes an in-place
    # reconfigure in run() reach every command's output.
    monkeypatch.setattr(cli, "app", lambda: cli.console.print("✓ Ready"))
    cli.run()
    stream.flush()
    assert "✓ Ready".encode() in stream.buffer.getvalue()
