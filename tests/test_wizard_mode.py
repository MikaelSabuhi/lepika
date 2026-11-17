from __future__ import annotations

import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fakes import Backends, FakeBackend, FakeResponse, FakeSystem, FakeUI, routes
from typer.testing import CliRunner

from lepika import (
    acquire,
    backend,
    cli,
    config,
    detect,
    engine,
    models,
    ollama,
    system,
    ui,
    wizard,
)
from lepika.errors import FriendlyError
from lepika.server import ServerBackend

runner = CliRunner()
DOCKER = detect.Machine("linux", "x86_64", "nvidia", 64.0)
NO_DOCKER = detect.Machine("macos", "arm64", "apple", 16.0)
CURATED = [models.CuratedModel(name="Small", ref="llama3.2:3b", min_ram_gb=6)]


def docker_installed() -> FakeSystem:
    """A machine whose PATH has Docker on it, and nothing else that is asked about."""
    return FakeSystem(which={"docker": "/usr/bin/docker"})


def started(
    info: detect.Machine,
    cfg: config.Config,
    after_engine: Callable[[], None] | None = None,
    **k: Any,
) -> str:
    """A backend start that runs the hook, which is what both real ones do."""
    if after_engine is not None:
        after_engine()
    return "http://localhost:3000"


def fake_for(mode: str) -> FakeBackend:
    """The fake the test installed for this mode."""
    picker = backend.backend_for
    assert isinstance(picker, Backends)
    found: FakeBackend = picker.by_mode[mode]
    return found


def test_choose_mode_is_silent_without_docker() -> None:
    """Never ask a non-Docker user about Docker — not even to say no."""
    asked: list[str] = []

    def ask(*a: Any, **k: Any) -> str:
        asked.append("?")
        return "2"

    machine = FakeSystem()
    assert wizard.choose_mode("express", sys=machine, ui=FakeUI(reader=ask)) == "express"
    assert asked == []
    assert machine.looked_up == ["docker"]


def test_choose_mode_asks_when_docker_is_present_and_defaults_to_current() -> None:
    assert (
        wizard.choose_mode("express", sys=docker_installed(), ui=FakeUI(reader=lambda *a, **k: ""))
        == "express"
    )
    assert (
        wizard.choose_mode("server", sys=docker_installed(), ui=FakeUI(reader=lambda *a, **k: ""))
        == "server"
    )
    assert (
        wizard.choose_mode("express", sys=docker_installed(), ui=FakeUI(reader=lambda *a, **k: "2"))
        == "server"
    )
    assert (
        wizard.choose_mode("server", sys=docker_installed(), ui=FakeUI(reader=lambda *a, **k: "1"))
        == "express"
    )


def _no_terminal(*a: Any, **k: Any) -> str:
    raise EOFError("stdin is not a terminal")


def test_choose_mode_keeps_the_current_mode_without_a_terminal(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        wizard.choose_mode("server", sys=docker_installed(), ui=FakeUI(reader=_no_terminal))
        == "server"
    )
    out = capsys.readouterr().out
    assert "No terminal to ask — staying with Server." in out
    assert "using 1" not in out and "using 2" not in out
    assert (
        wizard.choose_mode("express", sys=docker_installed(), ui=FakeUI(reader=_no_terminal))
        == "express"
    )
    out = capsys.readouterr().out
    assert "No terminal to ask — staying with Express." in out
    assert "using 1" not in out


def test_choose_mode_accepts_the_mode_by_name() -> None:
    assert (
        wizard.choose_mode(
            "express", sys=docker_installed(), ui=FakeUI(reader=lambda *a, **k: "server")
        )
        == "server"
    )
    assert (
        wizard.choose_mode(
            "server", sys=docker_installed(), ui=FakeUI(reader=lambda *a, **k: "Express")
        )
        == "express"
    )


def test_choose_mode_asks_again_once_then_keeps_the_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    answers = iter(["3", "x"])
    asked: list[str] = []

    def ask(prompt: str, **k: Any) -> str:
        asked.append(prompt)
        return next(answers)

    assert wizard.choose_mode("server", sys=docker_installed(), ui=FakeUI(reader=ask)) == "server"
    assert len(asked) == 2
    assert "Pick 1 or 2" in capsys.readouterr().out


def test_wizard_in_server_mode_uses_the_server_backend(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """Picking 🐳 must reach the Server backend, with the pull still between engine and UI."""
    events: list[str] = []

    def fake_server_start(
        info: detect.Machine,
        cfg: config.Config,
        after_engine: Callable[[], None] | None = None,
        **k: Any,
    ) -> str:
        if after_engine is not None:
            after_engine()
        events.append("server")
        return "http://localhost:3000"

    monkeypatch.setattr(detect, "detect", lambda **k: DOCKER)
    monkeypatch.setattr(system, "REAL", docker_installed())
    monkeypatch.setattr(models, "load_curated", lambda **k: CURATED)
    answers = iter(["2", "1"])  # mode, then model
    monkeypatch.setattr(ui, "REAL", FakeUI(reader=lambda *a, **k: next(answers)))
    monkeypatch.setattr(
        backend,
        "backend_for",
        Backends(
            server=FakeBackend(on_start=fake_server_start),
            express=FakeBackend(
                on_start=lambda *a, **k: pytest.fail("wrong backend"),
                # Leaving Express stops its stack: nothing of it runs here.
                on_stop=lambda info, cfg, **k: False,
            ),
        ),
    )
    monkeypatch.setattr(engine, "pull_model", lambda url, ref, **k: events.append("pull"))
    # Nor did LePika start an Ollama here: there is none to stop.
    monkeypatch.setattr(ollama, "stop_ollama", lambda os_name, url, **k: False)
    monkeypatch.setattr(cli, "_open_browser", lambda url: events.append("browser"))
    result = runner.invoke(cli.app, [])
    assert result.exit_code == 0, result.output
    assert events == ["pull", "server", "browser"]
    assert config.load().mode == "server"
    assert config.load().model == "llama3.2:3b"


def test_mode_flag_skips_the_question(monkeypatch: pytest.MonkeyPatch, isolated_home: Path) -> None:
    monkeypatch.setattr(detect, "detect", lambda **k: DOCKER)
    monkeypatch.setattr(system, "REAL", docker_installed())
    monkeypatch.setattr(models, "load_curated", lambda **k: CURATED)
    monkeypatch.setattr(ui, "REAL", FakeUI(reader=lambda *a, **k: "1"))  # only the model question
    monkeypatch.setattr(
        backend,
        "backend_for",
        Backends(
            server=FakeBackend(on_start=started),
            # Leaving Express stops its stack: nothing of it runs here.
            express=FakeBackend(on_stop=lambda info, cfg, **k: False),
        ),
    )
    # Nor did LePika start an Ollama here: there is none to stop.
    monkeypatch.setattr(ollama, "stop_ollama", lambda os_name, url, **k: False)
    monkeypatch.setattr(engine, "pull_model", lambda url, ref, **k: None)
    monkeypatch.setattr(cli, "_open_browser", lambda url: None)
    result = runner.invoke(cli.app, ["--mode", "server"])
    assert result.exit_code == 0, result.output
    assert config.load().mode == "server"


def test_mode_server_without_docker_is_friendly_not_an_install_prompt(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    machine = FakeSystem()  # no Docker on the PATH
    monkeypatch.setattr(detect, "detect", lambda **k: NO_DOCKER)
    monkeypatch.setattr(system, "REAL", machine)
    result = runner.invoke(cli.app, ["--mode", "server"])
    assert result.exit_code != 0
    assert "Docker" in str(result.exception)
    assert "lepika --mode express" in result.exception.fix  # type: ignore[union-attr]
    # Nothing is run to find that out: no `docker info` for a Docker that isn't there.
    assert machine.ran == []


def test_mode_rejects_anything_but_express_or_server(isolated_home: Path) -> None:
    result = runner.invoke(cli.app, ["--mode", "kubernetes"])
    assert result.exit_code != 0


def test_dry_run_in_server_mode_describes_compose(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    config.save(config.Config(mode="server"))
    monkeypatch.setattr(detect, "detect", lambda **k: DOCKER)
    machine = docker_installed()
    monkeypatch.setattr(system, "REAL", machine)
    monkeypatch.setattr(models, "load_curated", lambda **k: CURATED)
    answers = iter(["", "1"])  # bare Enter keeps the saved mode; then the model
    monkeypatch.setattr(ui, "REAL", FakeUI(reader=lambda *a, **k: next(answers)))
    monkeypatch.setattr(
        backend,
        "backend_for",
        Backends(
            server=FakeBackend(on_start=lambda *a, **k: pytest.fail("--dry-run must start nothing"))
        ),
    )
    result = runner.invoke(cli.app, ["--dry-run"])
    assert result.exit_code == 0, result.output
    assert "docker compose" in result.output
    assert config.load().mode == "server"
    # The one command a dry run may run is the read that sizes the models.
    assert machine.ran == [["docker", "info", "--format", "{{.MemTotal}}"]]
    assert machine.called == [] and machine.streamed == [] and machine.spawned == []
    # "would:" means would: a dry run must not even lay down the stack directory.
    assert not (isolated_home / "stack").exists()


def test_mode_with_a_subcommand_is_a_usage_error(isolated_home: Path) -> None:
    """Accepting it silently would look like `up` had just switched modes."""
    result = runner.invoke(cli.app, ["--mode", "server", "up"])
    assert result.exit_code != 0
    assert config.load().mode == "express"


@pytest.fixture()
def switching(monkeypatch: pytest.MonkeyPatch, isolated_home: Path) -> list[str]:
    """A Docker machine where both backends start silently and every stop is recorded."""
    stops: list[str] = []
    monkeypatch.setattr(detect, "detect", lambda **k: DOCKER)
    monkeypatch.setattr(system, "REAL", docker_installed())
    monkeypatch.setattr(models, "load_curated", lambda **k: CURATED)
    monkeypatch.setattr(
        backend,
        "backend_for",
        Backends(
            express=FakeBackend(
                on_start=started,
                on_stop=lambda info, cfg, **k: stops.append("express") or True,
            ),
            server=FakeBackend(
                on_start=started,
                on_stop=lambda info, cfg, **k: stops.append("server") or True,
            ),
        ),
    )
    # No native Ollama of LePika's runs; a test about that stop replaces this.
    monkeypatch.setattr(ollama, "stop_ollama", lambda os_name, url, **k: False)
    monkeypatch.setattr(engine, "pull_model", lambda url, ref, **k: None)
    monkeypatch.setattr(cli, "_open_browser", lambda url: None)
    return stops


def test_leaving_server_mode_stops_the_server_stack(
    switching: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both modes bind the same port: the abandoned stack does not linger, it answers."""
    config.save(config.Config(mode="server"))
    answers = iter(["1", "1"])  # Express, then the model
    monkeypatch.setattr(ui, "REAL", FakeUI(reader=lambda *a, **k: next(answers)))
    result = runner.invoke(cli.app, [])
    assert result.exit_code == 0, result.output
    assert switching == ["server"]
    assert config.load().mode == "express"
    # Stopped with the old mode's backend, started with the new one's.
    assert fake_for("express").started and not fake_for("server").started


def test_leaving_express_mode_stops_the_express_stack(
    switching: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    answers = iter(["2", "1"])  # Server, then the model
    monkeypatch.setattr(ui, "REAL", FakeUI(reader=lambda *a, **k: next(answers)))
    result = runner.invoke(cli.app, [])
    assert result.exit_code == 0, result.output
    assert switching == ["express"]
    assert config.load().mode == "server"
    # Stopped with the old mode's backend, started with the new one's.
    assert fake_for("server").started and not fake_for("express").started


def test_staying_in_the_same_mode_stops_nothing(
    switching: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A re-run of the wizard is not a reason to tear down a working stack."""
    answers = iter(["1", "1"])
    monkeypatch.setattr(ui, "REAL", FakeUI(reader=lambda *a, **k: next(answers)))
    assert runner.invoke(cli.app, []).exit_code == 0
    assert switching == []


def test_dry_run_switches_nothing_off(
    switching: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ui, "REAL", FakeUI(reader=lambda *a, **k: "1"))  # only the model question
    result = runner.invoke(cli.app, ["--dry-run", "--mode", "server"])
    assert result.exit_code == 0, result.output
    assert switching == []


def test_failed_switch_to_server_removes_the_containers_it_created(
    switching: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The config still says express after a failed switch, so `lepika down` cannot
    see what `compose up` created — the error path has to remove it itself."""

    def failing_start(*a: Any, **k: Any) -> str:
        raise FriendlyError("`docker compose up` failed.", "Run `lepika logs`.")

    monkeypatch.setattr(fake_for("server"), "on_start", failing_start)
    answers = iter(["2", "1"])  # Server, then the model
    monkeypatch.setattr(ui, "REAL", FakeUI(reader=lambda *a, **k: next(answers)))
    result = runner.invoke(cli.app, [])
    assert result.exit_code != 0
    # leave_mode stopped Express, then the error path swept the half-created stack.
    assert switching == ["express", "server"]
    assert config.load().mode == "express"


def test_failed_start_while_already_in_server_mode_leaves_the_stack_alone(
    switching: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No switch, no invisibility: `lepika down` sees these containers, and a wizard
    re-run that fails must not tear down what was running before it."""
    config.save(config.Config(mode="server"))

    def failing_start(*a: Any, **k: Any) -> str:
        raise FriendlyError("`docker compose up` failed.", "Run `lepika logs`.")

    monkeypatch.setattr(fake_for("server"), "on_start", failing_start)
    answers = iter(["2", "1"])
    monkeypatch.setattr(ui, "REAL", FakeUI(reader=lambda *a, **k: next(answers)))
    result = runner.invoke(cli.app, [])
    assert result.exit_code != 0
    assert switching == []


def test_failure_after_the_mode_was_saved_keeps_the_visible_stack(
    switching: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once the engine came up the hook saved mode=server, so the containers are
    visible to `lepika down` — a later failure (OpenWebUI timeout) sweeps nothing."""

    def engine_up_then_webui_times_out(
        info: detect.Machine,
        cfg: config.Config,
        after_engine: Callable[[], None] | None = None,
        **k: Any,
    ) -> str:
        if after_engine is not None:
            after_engine()
        raise FriendlyError("OpenWebUI did not become ready within 180s.", "Run `lepika logs`.")

    monkeypatch.setattr(fake_for("server"), "on_start", engine_up_then_webui_times_out)
    answers = iter(["2", "1"])
    monkeypatch.setattr(ui, "REAL", FakeUI(reader=lambda *a, **k: next(answers)))
    result = runner.invoke(cli.app, [])
    assert result.exit_code != 0
    assert switching == ["express"]
    assert config.load().mode == "server"


def test_leaving_express_mode_stops_the_ollama_lepika_started(
    switching: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Server's pre-flight refuses a port 11434 that our own native Ollama still holds."""
    config.save(config.Config(mode="express", engine_key="s3cret"))
    seen: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        ollama,
        "stop_ollama",
        lambda os_name, url, key="", **k: bool(seen.append((os_name, url, key))) or True,
    )
    answers = iter(["2", "1"])  # Server, then the model
    monkeypatch.setattr(ui, "REAL", FakeUI(reader=lambda *a, **k: next(answers)))
    result = runner.invoke(cli.app, [])
    assert result.exit_code == 0, result.output
    # The key goes with it: a keyed engine answers 401 unauthenticated, which the
    # probe would read as "already down" and drop the pid file without stopping it.
    assert seen == [("linux", config.DEFAULT_ENGINE_URL, "s3cret")]
    assert "Stopped the Ollama LePika had started" in result.output


def test_servers_port_check_asks_again_after_the_switch_stopped_the_ollama(
    switching: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Leaving Express stops the Ollama LePika started; Server's pre-flight must then
    find port 11434 free. An answer taken before the stop made it refuse an Ollama that
    was already gone, so the port is asked about when the check runs, and only then."""
    events: list[str] = []
    native = {"running": True}

    def engine(request: urllib.request.Request) -> FakeResponse:
        answering = native["running"] or any("up" in cmd for cmd in machine.called)
        events.append("engine up" if answering else "engine refused")
        if not answering:
            raise OSError("connection refused")
        return FakeResponse(b"{}")

    webui = routes({"http://127.0.0.1:3000/health": FakeResponse(b"")})
    machine = FakeSystem(
        which={"docker": "/usr/bin/docker"},
        urls=lambda request: (
            engine(request)
            if request.full_url == "http://127.0.0.1:11434/api/version"
            else webui(request)
        ),
    )
    monkeypatch.setattr(system, "REAL", machine)

    def stop(os_name: str, url: str, **k: Any) -> bool:
        events.append("stop")
        native["running"] = False
        return True

    monkeypatch.setattr(ollama, "stop_ollama", stop)
    fakes = backend.backend_for
    monkeypatch.setattr(
        backend,
        "backend_for",
        lambda cfg, *, sys, ui: (
            ServerBackend(sys=sys, ui=ui) if cfg.mode == "server" else fakes(cfg, sys=sys, ui=ui)
        ),
    )
    monkeypatch.setattr(acquire, "acquire", lambda info, cfg, ref, **k: ref.raw)
    answers = iter(["2", "1"])  # Server, then the model
    monkeypatch.setattr(ui, "REAL", FakeUI(reader=lambda *a, **k: next(answers)))
    result = runner.invoke(cli.app, [])
    assert result.exit_code == 0, result.output
    # The pre-flight's probe comes after the stop and sees the port free; the next one
    # is the wait for the stack's own engine once `compose up` has run.
    assert events == ["stop", "engine refused", "engine up"]
    # Free, so there was no container of ours to ask about.
    assert not any("ps" in cmd for cmd in machine.ran)


def test_an_ollama_lepika_did_not_start_is_left_alone_and_unannounced(
    switching: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No pid file means Homebrew or systemd owns it: say nothing, stop nothing."""
    monkeypatch.setattr(ollama, "stop_ollama", lambda os_name, url, **k: False)
    answers = iter(["2", "1"])
    monkeypatch.setattr(ui, "REAL", FakeUI(reader=lambda *a, **k: next(answers)))
    result = runner.invoke(cli.app, [])
    assert result.exit_code == 0, result.output
    assert "Stopped the Ollama" not in result.output


def test_leaving_express_mode_never_stops_a_remote_engine(
    switching: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule 9: an engine someone else runs is never started or stopped by us."""
    config.save(config.Config(mode="express", engine_managed=False))
    monkeypatch.setattr(
        ollama, "stop_ollama", lambda *a, **k: pytest.fail("must not touch a remote engine")
    )
    answers = iter(["2", "1"])
    monkeypatch.setattr(ui, "REAL", FakeUI(reader=lambda *a, **k: next(answers)))
    assert runner.invoke(cli.app, []).exit_code == 0


def test_leaving_server_mode_does_not_stop_a_native_ollama(
    switching: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stack we are leaving is the containers; a native Ollama was never ours here."""
    config.save(config.Config(mode="server"))
    monkeypatch.setattr(
        ollama, "stop_ollama", lambda *a, **k: pytest.fail("Server mode started no native Ollama")
    )
    answers = iter(["1", "1"])  # Express, then the model
    monkeypatch.setattr(ui, "REAL", FakeUI(reader=lambda *a, **k: next(answers)))
    assert runner.invoke(cli.app, []).exit_code == 0


def test_the_mode_is_saved_only_once_the_engine_is_up(
    switching: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-flight that refuses must not leave a config claiming a mode that never started."""

    def refuse(info: detect.Machine, cfg: config.Config, **k: Any) -> str:
        raise FriendlyError("Ollama is already running natively on port 11434.", "Stop it.")

    config.save(config.Config(mode="express", model="previous:model"))
    monkeypatch.setattr(ollama, "stop_ollama", lambda os_name, url, **k: False)
    monkeypatch.setattr(fake_for("server"), "on_start", refuse)
    answers = iter(["2", "1"])  # Server, then the model
    monkeypatch.setattr(ui, "REAL", FakeUI(reader=lambda *a, **k: next(answers)))
    result = runner.invoke(cli.app, [])
    assert result.exit_code != 0
    assert config.load().mode == "express"
    assert config.load().model == "previous:model"


def test_a_backend_too_broken_to_stop_does_not_trap_the_user_in_it(
    switching: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Docker being down is usually *why* someone is leaving Server mode."""

    def boom(info: detect.Machine, cfg: config.Config, **k: Any) -> bool:
        raise FriendlyError("Docker is installed but not running.", "Start Docker Desktop.")

    config.save(config.Config(mode="server"))
    monkeypatch.setattr(fake_for("server"), "on_stop", boom)
    answers = iter(["1", "1"])
    monkeypatch.setattr(ui, "REAL", FakeUI(reader=lambda *a, **k: next(answers)))
    result = runner.invoke(cli.app, [])
    assert result.exit_code == 0, result.output
    assert "not running" in result.output
    assert config.load().mode == "express"


def test_a_ui_that_will_not_stop_still_lets_the_engine_stop(
    switching: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two stops, two reports: one failure must not swallow the other's work.

    Leaving Express with a wedged OpenWebUI would otherwise skip `stop_ollama`
    entirely, and Server's pre-flight then refuses the port 11434 it still holds.
    """

    def boom(info: detect.Machine, cfg: config.Config, **k: Any) -> bool:
        raise FriendlyError("OpenWebUI on port 3000 is still answering.", "Stop it yourself.")

    monkeypatch.setattr(fake_for("express"), "on_stop", boom)
    monkeypatch.setattr(ollama, "stop_ollama", lambda os_name, url, **k: True)
    answers = iter(["2", "1"])  # Server, then the model
    monkeypatch.setattr(ui, "REAL", FakeUI(reader=lambda *a, **k: next(answers)))
    result = runner.invoke(cli.app, [])
    assert result.exit_code == 0, result.output
    assert "still answering" in result.output
    assert "Stopped the Ollama LePika had started" in result.output


def test_an_engine_that_will_not_stop_still_reports_the_ui_stop(
    switching: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mirror image: an Ollama that will not die is its own yellow line."""

    def boom(os_name: str, url: str, **k: Any) -> bool:
        raise FriendlyError(f"Ollama at {url} is still answering.", "Stop it yourself.")

    monkeypatch.setattr(ollama, "stop_ollama", boom)
    answers = iter(["2", "1"])  # Server, then the model
    monkeypatch.setattr(ui, "REAL", FakeUI(reader=lambda *a, **k: next(answers)))
    result = runner.invoke(cli.app, [])
    assert result.exit_code == 0, result.output
    assert switching == ["express"]
    assert "still answering" in result.output
