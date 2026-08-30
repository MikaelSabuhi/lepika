from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from lepika import cli, config, detect, engine, express, models, server, wizard
from lepika.errors import FriendlyError

runner = CliRunner()
DOCKER = detect.SystemInfo("linux", "x86_64", "nvidia", 64.0, True, False, False)
NO_DOCKER = detect.SystemInfo("macos", "arm64", "apple", 16.0, False, True, True)
CURATED = [models.CuratedModel(name="Small", ref="llama3.2:3b", min_ram_gb=6)]


def started(
    info: detect.SystemInfo,
    cfg: config.Config,
    after_engine: Callable[[], None] | None = None,
    **k: Any,
) -> str:
    """A backend start that runs the hook, which is what both real ones do."""
    if after_engine is not None:
        after_engine()
    return "http://localhost:3000"


def test_choose_mode_is_silent_without_docker() -> None:
    """Never ask a non-Docker user about Docker — not even to say no."""
    asked: list[str] = []

    def ask(*a: Any, **k: Any) -> str:
        asked.append("?")
        return "2"

    assert wizard.choose_mode(NO_DOCKER, "express", ask=ask) == "express"
    assert asked == []


def test_choose_mode_asks_when_docker_is_present_and_defaults_to_current() -> None:
    assert wizard.choose_mode(DOCKER, "express", ask=lambda *a, **k: "") == "express"
    assert wizard.choose_mode(DOCKER, "server", ask=lambda *a, **k: "") == "server"
    assert wizard.choose_mode(DOCKER, "express", ask=lambda *a, **k: "2") == "server"
    assert wizard.choose_mode(DOCKER, "server", ask=lambda *a, **k: "1") == "express"


def test_wizard_in_server_mode_uses_the_server_backend(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """Picking 🐳 must reach `server.start_stack`, with the pull still between engine and UI."""
    events: list[str] = []

    def fake_server_start(
        info: detect.SystemInfo,
        cfg: config.Config,
        after_engine: Callable[[], None] | None = None,
        **k: Any,
    ) -> str:
        if after_engine is not None:
            after_engine()
        events.append("server")
        return "http://localhost:3000"

    monkeypatch.setattr(detect, "detect", lambda **k: DOCKER)
    monkeypatch.setattr(models, "load_curated", lambda **k: CURATED)
    answers = iter(["2", "1"])  # mode, then model
    monkeypatch.setattr(wizard, "_ask", lambda *a, **k: next(answers))
    monkeypatch.setattr(server, "start_stack", fake_server_start)
    monkeypatch.setattr(engine, "pull_model", lambda url, ref, **k: events.append("pull"))
    monkeypatch.setattr(express, "start_stack", lambda *a, **k: pytest.fail("wrong backend"))
    monkeypatch.setattr(cli, "_open_browser", lambda url: events.append("browser"))
    result = runner.invoke(cli.app, [])
    assert result.exit_code == 0, result.output
    assert events == ["pull", "server", "browser"]
    assert config.load().mode == "server"
    assert config.load().model == "llama3.2:3b"


def test_mode_flag_skips_the_question(monkeypatch: pytest.MonkeyPatch, isolated_home: Path) -> None:
    monkeypatch.setattr(detect, "detect", lambda **k: DOCKER)
    monkeypatch.setattr(models, "load_curated", lambda **k: CURATED)
    monkeypatch.setattr(wizard, "_ask", lambda *a, **k: "1")  # only the model question
    monkeypatch.setattr(server, "start_stack", started)
    monkeypatch.setattr(engine, "pull_model", lambda url, ref, **k: None)
    monkeypatch.setattr(cli, "_open_browser", lambda url: None)
    result = runner.invoke(cli.app, ["--mode", "server"])
    assert result.exit_code == 0, result.output
    assert config.load().mode == "server"


def test_mode_server_without_docker_is_friendly_not_an_install_prompt(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    monkeypatch.setattr(detect, "detect", lambda **k: NO_DOCKER)
    result = runner.invoke(cli.app, ["--mode", "server"])
    assert result.exit_code != 0
    assert "Docker" in str(result.exception)


def test_mode_rejects_anything_but_express_or_server(isolated_home: Path) -> None:
    result = runner.invoke(cli.app, ["--mode", "kubernetes"])
    assert result.exit_code != 0


def test_dry_run_in_server_mode_describes_compose(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    config.save(config.Config(mode="server"))
    monkeypatch.setattr(detect, "detect", lambda **k: DOCKER)
    monkeypatch.setattr(models, "load_curated", lambda **k: CURATED)
    answers = iter(["", "1"])  # bare Enter keeps the saved mode; then the model
    monkeypatch.setattr(wizard, "_ask", lambda *a, **k: next(answers))
    monkeypatch.setattr(
        server, "start_stack", lambda *a, **k: pytest.fail("--dry-run must start nothing")
    )
    result = runner.invoke(cli.app, ["--dry-run"])
    assert result.exit_code == 0, result.output
    assert "docker compose" in result.output
    assert config.load().mode == "server"
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
    monkeypatch.setattr(models, "load_curated", lambda **k: CURATED)
    monkeypatch.setattr(express, "stop", lambda info, cfg, **k: stops.append("express") or True)
    monkeypatch.setattr(server, "stop", lambda info, cfg, **k: stops.append("server") or True)
    monkeypatch.setattr(express, "start_stack", started)
    monkeypatch.setattr(server, "start_stack", started)
    monkeypatch.setattr(engine, "pull_model", lambda url, ref, **k: None)
    monkeypatch.setattr(cli, "_open_browser", lambda url: None)
    return stops


def test_leaving_server_mode_stops_the_server_stack(
    switching: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both modes bind the same port: the abandoned stack does not linger, it answers."""
    config.save(config.Config(mode="server"))
    answers = iter(["1", "1"])  # Express, then the model
    monkeypatch.setattr(wizard, "_ask", lambda *a, **k: next(answers))
    result = runner.invoke(cli.app, [])
    assert result.exit_code == 0, result.output
    assert switching == ["server"]
    assert config.load().mode == "express"


def test_leaving_express_mode_stops_the_express_stack(
    switching: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    answers = iter(["2", "1"])  # Server, then the model
    monkeypatch.setattr(wizard, "_ask", lambda *a, **k: next(answers))
    result = runner.invoke(cli.app, [])
    assert result.exit_code == 0, result.output
    assert switching == ["express"]
    assert config.load().mode == "server"


def test_staying_in_the_same_mode_stops_nothing(
    switching: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A re-run of the wizard is not a reason to tear down a working stack."""
    answers = iter(["1", "1"])
    monkeypatch.setattr(wizard, "_ask", lambda *a, **k: next(answers))
    assert runner.invoke(cli.app, []).exit_code == 0
    assert switching == []


def test_dry_run_switches_nothing_off(
    switching: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wizard, "_ask", lambda *a, **k: "1")  # only the model question
    result = runner.invoke(cli.app, ["--dry-run", "--mode", "server"])
    assert result.exit_code == 0, result.output
    assert switching == []


def test_leaving_express_mode_stops_the_ollama_lepika_started(
    switching: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Server's pre-flight refuses a port 11434 that our own native Ollama still holds."""
    config.save(config.Config(mode="express", engine_key="s3cret"))
    seen: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        express,
        "stop_ollama",
        lambda os_name, url, key="", **k: bool(seen.append((os_name, url, key))) or True,
    )
    answers = iter(["2", "1"])  # Server, then the model
    monkeypatch.setattr(wizard, "_ask", lambda *a, **k: next(answers))
    result = runner.invoke(cli.app, [])
    assert result.exit_code == 0, result.output
    # The key goes with it: a keyed engine answers 401 unauthenticated, which the
    # probe would read as "already down" and drop the pid file without stopping it.
    assert seen == [("linux", config.DEFAULT_ENGINE_URL, "s3cret")]
    assert "Stopped the Ollama LePika had started" in result.output


def test_server_start_sees_the_ollama_as_stopped_after_the_switch(
    switching: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Detection ran before the stop: handing the old snapshot to Server's pre-flight
    made it refuse port 11434 for an Ollama that was already gone (E2E finding F13)."""
    monkeypatch.setattr(
        detect, "detect", lambda **k: dataclasses.replace(DOCKER, ollama_running=True)
    )
    monkeypatch.setattr(express, "stop_ollama", lambda os_name, url, **k: True)
    seen: list[detect.SystemInfo] = []

    def start(info: detect.SystemInfo, cfg: config.Config, **k: Any) -> str:
        seen.append(info)
        return started(info, cfg, **k)

    monkeypatch.setattr(server, "start_stack", start)
    answers = iter(["2", "1"])  # Server, then the model
    monkeypatch.setattr(wizard, "_ask", lambda *a, **k: next(answers))
    result = runner.invoke(cli.app, [])
    assert result.exit_code == 0, result.output
    assert [info.ollama_running for info in seen] == [False]


def test_an_ollama_lepika_did_not_start_is_left_alone_and_unannounced(
    switching: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No pid file means Homebrew or systemd owns it: say nothing, stop nothing."""
    monkeypatch.setattr(express, "stop_ollama", lambda os_name, url, **k: False)
    answers = iter(["2", "1"])
    monkeypatch.setattr(wizard, "_ask", lambda *a, **k: next(answers))
    result = runner.invoke(cli.app, [])
    assert result.exit_code == 0, result.output
    assert "Stopped the Ollama" not in result.output


def test_leaving_express_mode_never_stops_a_remote_engine(
    switching: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule 9: an engine someone else runs is never started or stopped by us."""
    config.save(config.Config(mode="express", engine_managed=False))
    monkeypatch.setattr(
        express, "stop_ollama", lambda *a, **k: pytest.fail("must not touch a remote engine")
    )
    answers = iter(["2", "1"])
    monkeypatch.setattr(wizard, "_ask", lambda *a, **k: next(answers))
    assert runner.invoke(cli.app, []).exit_code == 0


def test_leaving_server_mode_does_not_stop_a_native_ollama(
    switching: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stack we are leaving is the containers; a native Ollama was never ours here."""
    config.save(config.Config(mode="server"))
    monkeypatch.setattr(
        express, "stop_ollama", lambda *a, **k: pytest.fail("Server mode started no native Ollama")
    )
    answers = iter(["1", "1"])  # Express, then the model
    monkeypatch.setattr(wizard, "_ask", lambda *a, **k: next(answers))
    assert runner.invoke(cli.app, []).exit_code == 0


def test_the_mode_is_saved_only_once_the_engine_is_up(
    switching: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-flight that refuses must not leave a config claiming a mode that never started."""

    def refuse(info: detect.SystemInfo, cfg: config.Config, **k: Any) -> str:
        raise FriendlyError("Ollama is already running natively on port 11434.", "Stop it.")

    config.save(config.Config(mode="express", model="previous:model"))
    monkeypatch.setattr(express, "stop_ollama", lambda os_name, url, **k: False)
    monkeypatch.setattr(server, "start_stack", refuse)
    answers = iter(["2", "1"])  # Server, then the model
    monkeypatch.setattr(wizard, "_ask", lambda *a, **k: next(answers))
    result = runner.invoke(cli.app, [])
    assert result.exit_code != 0
    assert config.load().mode == "express"
    assert config.load().model == "previous:model"


def test_a_backend_too_broken_to_stop_does_not_trap_the_user_in_it(
    switching: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Docker being down is usually *why* someone is leaving Server mode."""

    def boom(info: detect.SystemInfo, cfg: config.Config, **k: Any) -> bool:
        raise FriendlyError("Docker is installed but not running.", "Start Docker Desktop.")

    config.save(config.Config(mode="server"))
    monkeypatch.setattr(server, "stop", boom)
    answers = iter(["1", "1"])
    monkeypatch.setattr(wizard, "_ask", lambda *a, **k: next(answers))
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

    def boom(info: detect.SystemInfo, cfg: config.Config, **k: Any) -> bool:
        raise FriendlyError("OpenWebUI on port 3000 is still answering.", "Stop it yourself.")

    monkeypatch.setattr(express, "stop", boom)
    monkeypatch.setattr(express, "stop_ollama", lambda os_name, url, **k: True)
    answers = iter(["2", "1"])  # Server, then the model
    monkeypatch.setattr(wizard, "_ask", lambda *a, **k: next(answers))
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

    monkeypatch.setattr(express, "stop_ollama", boom)
    answers = iter(["2", "1"])  # Server, then the model
    monkeypatch.setattr(wizard, "_ask", lambda *a, **k: next(answers))
    result = runner.invoke(cli.app, [])
    assert result.exit_code == 0, result.output
    assert switching == ["express"]
    assert "still answering" in result.output
