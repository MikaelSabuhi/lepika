from __future__ import annotations

import importlib.metadata
import io
import json
import re
import sys
import urllib.error
from email.message import Message
from pathlib import Path
from typing import Any

import pytest
from fakes import Backends, FakeBackend, FakeResponse, FakeSystem, FakeUI, routes
from typer.testing import CliRunner

from lepika import backend, cli, config, detect, health, models, system, ui
from lepika.cli import app
from lepika.errors import FriendlyError

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
    monkeypatch.setattr(cli, "app", lambda **k: ui.REAL.say("✓ Ready"))
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
    monkeypatch.setattr(cli, "app", lambda **k: sys.stderr.write("✗ nope"))
    cli.run()
    stream.flush()
    assert stream.buffer.getvalue() == "✗ nope".encode()


def _raise(exc: BaseException) -> Any:
    def app(**k: Any) -> None:
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


def test_run_reports_an_unreadable_file_by_name_and_folder(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], isolated_home: Path
) -> None:
    """Reads fail too, and the file may be the user's own: name it and its folder."""
    error = PermissionError(13, "Permission denied", "/data/models/config.json")
    monkeypatch.setattr(cli, "app", _raise(error))
    with pytest.raises(SystemExit) as exit_info:
        cli.run()
    assert exit_info.value.code == 1
    err = _flat(capsys.readouterr().err)
    assert "✗ Could not access /data/models/config.json: Permission denied" in err
    folder = Path("/data/models")
    assert f"→ Check the permissions on {folder} and that the disk is not full." in err
    assert str(isolated_home) not in err
    assert "Traceback" not in err


def test_run_reports_a_missing_command_as_such(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bare name is a program that is not installed, not a file with bad permissions."""
    monkeypatch.setattr(
        cli, "app", _raise(FileNotFoundError(2, "No such file or directory", "sudo"))
    )
    with pytest.raises(SystemExit) as exit_info:
        cli.run()
    assert exit_info.value.code == 1
    err = _flat(capsys.readouterr().err)
    assert "✗ Command not found: sudo" in err
    assert "→ Install sudo or run `lepika doctor` for setup help." in err
    assert "Could not access" not in err


def test_run_reports_a_dangling_file_as_a_file_not_a_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`model import .` names files relative to the folder: a bare name that exists as a
    path, even a dangling link, is a file."""
    name = "model-00001-of-00002.safetensors"
    try:
        (tmp_path / name).symlink_to(tmp_path / "gone.safetensors")
    except OSError:
        pytest.skip("this platform does not let the test create a symlink")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "app", _raise(FileNotFoundError(2, "No such file or directory", name)))
    with pytest.raises(SystemExit) as exit_info:
        cli.run()
    assert exit_info.value.code == 1
    err = _flat(capsys.readouterr().err)
    assert f"✗ Could not access {name}: No such file or directory" in err
    assert "Command not found" not in err


@pytest.mark.parametrize(
    "error",
    [
        OSError("boom"),
        urllib.error.HTTPError("https://example.invalid/x", 500, "Server Error", Message(), None),
    ],
)
def test_run_treats_an_os_error_without_a_file_as_unexpected(
    error: OSError, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "app", _raise(error))
    with pytest.raises(SystemExit) as exit_info:
        cli.run()
    assert exit_info.value.code == 1
    err = _flat(capsys.readouterr().err)
    assert "unexpected error" in err
    assert "Could not access" not in err
    assert "Traceback" not in err


def test_run_without_a_terminal_is_two_lines_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    info = detect.Machine("macos", "arm64", "apple", 16.0)

    def no_terminal(*a: Any, **k: Any) -> str:
        raise EOFError("stdin is not a terminal")

    monkeypatch.setattr(sys, "argv", ["lepika"])
    monkeypatch.setattr(detect, "detect", lambda **k: info)
    monkeypatch.setattr(system, "REAL", FakeSystem())  # no Docker: no mode question
    monkeypatch.setattr(
        models,
        "load_curated",
        lambda **k: [models.CuratedModel(name="Small", ref="llama3.2:3b", min_ram_gb=6)],
    )
    monkeypatch.setattr(health, "webui_up", lambda port, **k: True)
    monkeypatch.setattr(ui, "REAL", FakeUI(reader=no_terminal))
    with pytest.raises(SystemExit) as exit_info:
        cli.run()
    assert exit_info.value.code == 1
    err = _flat(capsys.readouterr().err)
    assert "✗ Choosing a model needs an answer, and there is no terminal to ask." in err
    assert "Traceback" not in err


class _ReachedGlobal(BaseException):
    """A command reached a module global although it was handed its own `Deps`.

    A BaseException, like the suite's other refusals: no `except Exception` in `src/`
    may read it as a failed probe and carry on.
    """


class _Forbidden:
    """Stands in for a global no command may touch: any use of it raises."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, attr: str) -> Any:
        raise _ReachedGlobal(f"{self._name}.{attr}")

    def __call__(self, *a: Any, **k: Any) -> Any:
        raise _ReachedGlobal(self._name)


def _forbid_the_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(system, "REAL", _Forbidden("system.REAL"))
    monkeypatch.setattr(ui, "REAL", _Forbidden("ui.REAL"))
    monkeypatch.setattr(detect, "detect", _Forbidden("detect.detect"))
    monkeypatch.setattr(backend, "backend_for", _Forbidden("backend.backend_for"))


MAC = detect.Machine("macos", "arm64", "apple", 16.0)


def test_a_command_works_through_the_deps_it_is_handed(monkeypatch: pytest.MonkeyPatch) -> None:
    chosen = FakeBackend(url="http://localhost:3000")
    detected: list[detect.Machine] = []
    picked: list[str] = []

    def machine() -> detect.Machine:
        detected.append(MAC)
        return MAC

    def backends(cfg: config.Config) -> Any:
        picked.append(cfg.mode)
        return chosen

    deps = cli.Deps(sys=FakeSystem(), ui=FakeUI(), machine=machine, backends=backends)
    opened: list[str] = []
    monkeypatch.setattr(cli, "_open_browser", opened.append)
    _forbid_the_globals(monkeypatch)
    result = runner.invoke(cli.app, ["up"], obj=deps)
    assert result.exit_code == 0, result.output
    assert detected == [MAC]
    assert picked == ["express"]
    assert [(info, cfg.mode) for info, cfg, _ in chosen.started] == [(MAC, "express")]
    assert "Ready: http://localhost:3000" in result.output
    assert opened == ["http://localhost:3000"]


def test_a_model_command_is_handed_the_same_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """The `model` group is nested: what the root was handed must reach its commands."""
    tags = f"{config.DEFAULT_ENGINE_URL}/api/tags"
    machine = FakeSystem(
        urls=routes({tags: FakeResponse(b'{"models": [{"name": "qwen3:8b", "size": 5}]}')})
    )
    deps = cli.Deps(
        sys=machine,
        ui=FakeUI(),
        machine=_Forbidden("machine"),
        backends=_Forbidden("backends"),
    )
    _forbid_the_globals(monkeypatch)
    result = runner.invoke(cli.app, ["model", "list"], obj=deps)
    assert result.exit_code == 0, result.output
    assert [r.full_url for r in machine.requested] == [tags]
    assert "qwen3:8b" in result.output


def test_without_deps_the_root_builds_the_real_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[cli.Deps] = []
    real = cli.Deps.real

    def spy() -> cli.Deps:
        built.append(real())
        return built[-1]

    monkeypatch.setattr(cli.Deps, "real", spy)
    fake_sys, fake_ui = FakeSystem(), FakeUI()
    monkeypatch.setattr(system, "REAL", fake_sys)
    monkeypatch.setattr(ui, "REAL", fake_ui)
    monkeypatch.setattr(detect, "detect", lambda **k: MAC)
    seen: list[tuple[Any, Any]] = []

    def backend_for(cfg: config.Config, *, sys: Any, ui: Any) -> Any:
        seen.append((sys, ui))
        return FakeBackend()

    monkeypatch.setattr(backend, "backend_for", backend_for)
    monkeypatch.setattr(cli, "_open_browser", lambda url: None)
    result = runner.invoke(cli.app, ["up"])
    assert result.exit_code == 0, result.output
    assert len(built) == 1
    assert built[0].sys is fake_sys
    assert built[0].ui is fake_ui
    assert seen == [(fake_sys, fake_ui)]


def test_real_deps_look_their_globals_up_when_called(monkeypatch: pytest.MonkeyPatch) -> None:
    """Built before a test replaces a global, the real `Deps` still reaches the new one."""
    deps = cli.Deps.real()
    fake_sys, fake_ui = FakeSystem(), FakeUI()
    monkeypatch.setattr(system, "REAL", fake_sys)
    monkeypatch.setattr(ui, "REAL", fake_ui)
    asked: list[Any] = []

    def detect_fake(*, sys: Any) -> detect.Machine:
        asked.append(sys)
        return MAC

    chosen = FakeBackend()
    backends = Backends(express=chosen)
    monkeypatch.setattr(detect, "detect", detect_fake)
    monkeypatch.setattr(backend, "backend_for", backends)
    assert deps.machine() is MAC
    assert asked == [fake_sys]
    assert deps.backends(config.Config()) is chosen
    assert backends.selected == ["express"]


def test_run_hands_the_app_its_deps_and_reports_through_their_ui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    face = FakeUI()
    deps = cli.Deps(
        sys=FakeSystem(), ui=face, machine=_Forbidden("machine"), backends=_Forbidden("backends")
    )
    monkeypatch.setattr(cli.Deps, "real", lambda: deps)
    handed: list[Any] = []

    def app(**kwargs: Any) -> None:
        handed.append(kwargs.get("obj"))
        raise FriendlyError("It broke.", "Fix it.")

    monkeypatch.setattr(cli, "app", app)
    errors: list[Any] = []
    monkeypatch.setattr(face, "say_error", lambda *objects: errors.append(objects))
    monkeypatch.setattr(ui, "REAL", _Forbidden("ui.REAL"))
    with pytest.raises(SystemExit) as exit_info:
        cli.run()
    assert exit_info.value.code == 1
    assert handed == [deps]
    assert errors == [("[red]✗ It broke.[/red]",), ("[yellow]→ Fix it.[/yellow]",)]


def _help_words(output: str) -> str:
    """Help text without colour, borders or wrapping: what a reader sees, in order."""
    plain = re.sub(r"\x1b\[[0-9;]*m", "", output)
    return " ".join(re.sub(r"[│╭╮╰╯─]", " ", plain).split())


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (
            ["--help"],
            "Usage: root [OPTIONS] COMMAND [ARGS]... One command → local AI chat in your "
            "browser. Options --version Show version and exit. --dry-run Show what the "
            "wizard would do without doing it. --mode <str> express (default, no Docker) "
            "or server (docker compose). --help Show this message and exit. Commands up "
            "Start the local AI stack and open the browser. down Stop the stack (Express: "
            "OpenWebUI only; Server: every container, models kept). status Show what's "
            "running. logs Print the tail of LePika's logs (Server mode: the container "
            "logs too). doctor Diagnose the local setup. update Upgrade the engine and "
            "OpenWebUI to their latest versions. connect Use an engine running on another "
            "machine (or --local to stop doing so). expose Share the engine (and the chat "
            "UI) with your network behind a generated API key. model Add, list, or remove "
            "local models.",
        ),
        (
            ["model", "--help"],
            "Usage: root model [OPTIONS] COMMAND [ARGS]... Add, list, or remove local "
            "models. Options --help Show this message and exit. Commands add Download a "
            "model and make it the default. import Import safetensors weights already on "
            "disk into Ollama. list List downloaded models. rm Remove a downloaded model.",
        ),
    ],
)
def test_help_shows_no_context_parameter(args: list[str], expected: str) -> None:
    """Every command takes a `typer.Context`; none of that may surface as an option."""
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0
    assert _help_words(result.output) == expected
