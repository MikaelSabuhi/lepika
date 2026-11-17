from __future__ import annotations

from pathlib import Path

import pytest
from fakes import Backends, FakeBackend, FakeSystem
from typer.testing import CliRunner

from lepika import backend, cli, config, detect, health, paths, server, system, ui
from lepika.server import ServerBackend

runner = CliRunner()
DOCKER = detect.Machine("linux", "x86_64", "nvidia", 64.0)


@pytest.fixture()
def server_mode(monkeypatch: pytest.MonkeyPatch, isolated_home: Path) -> list[str]:
    """A configured Server-mode machine where every Express entry point is a failure."""
    calls: list[str] = []
    config.save(config.Config(mode="server", model="qwen3:8b"))
    monkeypatch.setattr(detect, "detect", lambda **k: DOCKER)
    monkeypatch.setattr(system, "REAL", FakeSystem(which={"docker": "/usr/bin/docker"}))
    docker_stack = FakeBackend(
        on_start=lambda info, cfg, **k: calls.append("start") or "http://localhost:3000",
        on_stop=lambda info, cfg, **k: calls.append("stop") or True,
        on_update=lambda info, cfg, **k: calls.append("update") or [],
        on_logs=lambda lines, **k: [("docker compose", "ollama-1 | ready")],
        # The notes `up` prints are the real backend's: what they say is under test here.
        on_notes=lambda info, cfg: ServerBackend(sys=system.REAL, ui=ui.REAL).notes(info, cfg),
    )
    monkeypatch.setattr(server, "gpu_note", lambda info, **k: None)
    # No Express backend at all: selecting one fails the test (`UnprovidedBackend`).
    monkeypatch.setattr(backend, "backend_for", Backends(server=docker_stack))
    monkeypatch.setattr(cli, "_open_browser", lambda url: calls.append("browser"))
    return calls


def test_up_uses_server_backend(server_mode: list[str]) -> None:
    result = runner.invoke(cli.app, ["up"])
    assert result.exit_code == 0, result.output
    assert server_mode == ["start", "browser"]
    assert "Server mode" in result.output


def test_down_status_logs_update_use_server_backend(
    server_mode: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert runner.invoke(cli.app, ["down"]).exit_code == 0
    assert runner.invoke(cli.app, ["update"]).exit_code == 0
    assert server_mode == ["stop", "update"]
    assert "ready" in runner.invoke(cli.app, ["logs"]).output
    monkeypatch.setattr(health, "api_up", lambda url, **k: True)
    monkeypatch.setattr(health, "webui_up", lambda port, **k: True)
    out = runner.invoke(cli.app, ["status"]).output
    assert "server" in out


def test_up_prints_the_gpu_note_when_docker_cannot_see_the_gpu(
    server_mode: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        server, "gpu_note", lambda info, **k: "NVIDIA GPU found, but Docker can't use it"
    )
    assert "can't use it" in runner.invoke(cli.app, ["up"]).output


def _docker_host(containers: str) -> FakeSystem:
    """Docker running, and compose listing these container ids for the stack."""
    every = server.compose_cmd(paths.stack_dir(), list(server.INACTIVE), gpu_overlay=False)
    return FakeSystem(
        which={"docker": "/usr/bin/docker"}, stdout={" ".join([*every, "ps"]): containers}
    )


def test_down_in_server_mode_says_stopped_when_it_removed_containers(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    config.save(config.Config(mode="server"))
    monkeypatch.setattr(detect, "detect", lambda **k: DOCKER)
    monkeypatch.setattr(system, "REAL", _docker_host("3f2a9c\n"))
    result = runner.invoke(cli.app, ["down"])
    assert result.exit_code == 0, result.output
    assert "✓ Stopped." in result.output


def test_down_in_server_mode_says_nothing_was_running_when_there_were_no_containers(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    config.save(config.Config(mode="server"))
    monkeypatch.setattr(detect, "detect", lambda **k: DOCKER)
    monkeypatch.setattr(system, "REAL", _docker_host(""))
    result = runner.invoke(cli.app, ["down"])
    assert result.exit_code == 0, result.output
    assert "Nothing was running." in result.output
    assert "Stopped" not in result.output


def test_up_without_docker_says_so_before_probing_docker(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """`gpu_note` shells out to `docker info`; with no docker binary that is a generic
    "Command not found" instead of the friendly Express hint."""
    from lepika.errors import FriendlyError

    config.save(config.Config(mode="server"))
    monkeypatch.setattr(detect, "detect", lambda **k: DOCKER)
    machine = FakeSystem()  # no Docker on the PATH
    monkeypatch.setattr(system, "REAL", machine)
    monkeypatch.setattr(
        server, "gpu_note", lambda info, **k: pytest.fail("probed Docker before checking it exists")
    )
    result = runner.invoke(cli.app, ["up"])
    assert result.exit_code != 0
    assert isinstance(result.exception, FriendlyError)
    assert "Express" in result.exception.fix
    assert machine.ran == []
