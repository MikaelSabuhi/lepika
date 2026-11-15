from __future__ import annotations

import importlib.metadata
import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from lepika import cli, detect, express, models, wizard
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


def _raise(exc: BaseException) -> Any:
    def app() -> None:
        raise exc

    return app


def _flat(text: str) -> str:
    """Rich wraps at the console width; compare with the line breaks folded away."""
    return " ".join(text.split())


def test_run_turns_an_unexpected_exception_into_two_lines_and_a_log_entry(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], isolated_home: Path
) -> None:
    monkeypatch.setattr(cli, "app", _raise(ValueError("bad value")))
    with pytest.raises(SystemExit) as exit_info:
        cli.run()
    assert exit_info.value.code == 1
    err = _flat(capsys.readouterr().err)
    assert "unexpected error: ValueError: bad value" in err
    assert "https://github.com/MikaelSabuhi/lepika/issues" in err
    assert "Details are in" in err
    assert "Traceback" not in err
    lines = (isolated_home / "logs" / "lepika.log").read_text().splitlines()
    crash = [json.loads(line) for line in lines if '"cli.crash"' in line]
    assert len(crash) == 1
    assert "Traceback" in crash[0]["traceback"]
    assert crash[0]["error"] == "ValueError('bad value')"


def test_crash_handler_survives_an_unwritable_log(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def no_log() -> Any:
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(cli.log, "get_logger", no_log)
    monkeypatch.setattr(cli, "app", _raise(RuntimeError("boom")))
    with pytest.raises(SystemExit) as exit_info:
        cli.run()
    assert exit_info.value.code == 1
    err = _flat(capsys.readouterr().err)
    assert "unexpected error: RuntimeError: boom" in err
    assert "Details are in" not in err
    assert "https://github.com/MikaelSabuhi/lepika/issues" in err
    assert "Traceback" not in err


def test_run_reports_an_unwritable_file_by_name(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A short home keeps the fix on one console line; the handler must not create it.
    monkeypatch.setenv("LEPIKA_HOME", "/x/home")
    error = PermissionError(13, "Permission denied", "/x/config.toml")
    monkeypatch.setattr(cli, "app", _raise(error))
    with pytest.raises(SystemExit) as exit_info:
        cli.run()
    assert exit_info.value.code == 1
    err = _flat(capsys.readouterr().err)
    assert "✗ Could not write /x/config.toml: Permission denied" in err
    home = Path("/x/home")
    assert f"→ Check that {home} is writable and the disk is not full." in err
    assert "Traceback" not in err


def test_run_exits_130_quietly_on_ctrl_c(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "app", _raise(KeyboardInterrupt()))
    with pytest.raises(SystemExit) as exit_info:
        cli.run()
    assert exit_info.value.code == 130
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_run_without_a_terminal_is_two_lines_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    info = detect.SystemInfo("macos", "arm64", "apple", 16.0, False, True, True)

    def no_terminal(*a: Any, **k: Any) -> str:
        raise EOFError("stdin is not a terminal")

    monkeypatch.setattr(sys, "argv", ["lepika"])
    monkeypatch.setattr(detect, "detect", lambda **k: info)
    monkeypatch.setattr(
        models,
        "load_curated",
        lambda **k: [models.CuratedModel(name="Small", ref="llama3.2:3b", min_ram_gb=6)],
    )
    monkeypatch.setattr(express, "webui_up", lambda port, **k: True)
    monkeypatch.setattr(wizard, "_ask", no_terminal)
    with pytest.raises(SystemExit) as exit_info:
        cli.run()
    assert exit_info.value.code == 1
    err = _flat(capsys.readouterr().err)
    assert "✗ Choosing a model needs an answer, and there is no terminal to ask." in err
    assert "Traceback" not in err
