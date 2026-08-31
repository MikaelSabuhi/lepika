from __future__ import annotations

import importlib.metadata
import io
import sys
from typing import Any

import pytest
from typer.testing import CliRunner

from lepika import cli
from lepika.cli import app

runner = CliRunner()


def test_version_flag_prints_name_and_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    # pyproject.toml is the only place the version lives, so read it back from the
    # installed metadata rather than pinning a literal that every release must edit.
    assert result.output.strip() == f"lepika {importlib.metadata.version('lepika')}"


def _cp1252_stream() -> io.TextIOWrapper:
    # What Python hands a Windows process whose stdout is a pipe: the ANSI code page.
    return io.TextIOWrapper(io.BytesIO(), encoding="cp1252", write_through=True)


class _Recording(io.TextIOWrapper):
    """A wrapper that remembers whether anyone reconfigured it."""

    def __init__(self, encoding: str) -> None:
        super().__init__(io.BytesIO(), encoding=encoding, write_through=True)
        self.reconfigured = False

    def reconfigure(self, **kwargs: Any) -> None:  # type: ignore[override]
        self.reconfigured = True
        super().reconfigure(**kwargs)


def test_utf8_stdio_reconfigures_a_non_utf8_wrapper() -> None:
    stream = _Recording("cp1252")
    cli._utf8_stdio([stream])
    stream.write("✓")
    stream.flush()
    assert stream.reconfigured is True
    assert stream.buffer.getvalue() == "✓".encode()


def test_utf8_stdio_leaves_a_utf8_wrapper_alone() -> None:
    stream = _Recording("utf-8")
    cli._utf8_stdio([stream])
    assert stream.reconfigured is False


def test_utf8_stdio_treats_utf8_spelled_without_the_dash_as_utf8() -> None:
    stream = _Recording("utf8")
    cli._utf8_stdio([stream])
    assert stream.reconfigured is False


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


def test_run_reconfigures_a_cp1252_stderr_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """Errors go to stderr; a piped stderr on Windows is cp1252 as well."""
    stream = _cp1252_stream()
    monkeypatch.setattr(sys, "stderr", stream)
    monkeypatch.setattr(sys, "stdout", _cp1252_stream())
    # No trailing newline: the wrapper translates "\n" to os.linesep, which is "\r\n" on
    # the Windows runner this test is really about.
    monkeypatch.setattr(cli, "app", lambda: sys.stderr.write("✗ nope"))
    cli.run()
    stream.flush()
    assert stream.buffer.getvalue() == "✗ nope".encode()
