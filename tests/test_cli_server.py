from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lepika import cli, config, detect, express, server

runner = CliRunner()
DOCKER = detect.SystemInfo("linux", "x86_64", "nvidia", 64.0, True, False, False)


@pytest.fixture()
def server_mode(monkeypatch: pytest.MonkeyPatch, isolated_home: Path) -> list[str]:
    """A configured Server-mode machine where every Express entry point is a failure."""
    calls: list[str] = []
    config.save(config.Config(mode="server", model="qwen3:8b"))
    monkeypatch.setattr(detect, "detect", lambda **k: DOCKER)
    monkeypatch.setattr(
        server,
        "start_stack",
        lambda info, cfg, **k: calls.append("start") or "http://localhost:3000",
    )
    monkeypatch.setattr(server, "stop", lambda info, cfg, **k: calls.append("stop") or True)
    monkeypatch.setattr(server, "update", lambda info, cfg, **k: calls.append("update"))
    monkeypatch.setattr(server, "logs", lambda lines, **k: [("docker compose", "ollama-1 | ready")])
    monkeypatch.setattr(server, "gpu_note", lambda info, **k: None)
    for name in ("start_stack", "stop", "update", "logs"):
        monkeypatch.setattr(
            express, name, lambda *a, **k: pytest.fail("express backend used in server mode")
        )
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
    monkeypatch.setattr(detect, "api_up", lambda url, **k: True)
    monkeypatch.setattr(express, "webui_up", lambda port, **k: True)
    out = runner.invoke(cli.app, ["status"]).output
    assert "server" in out


def test_up_prints_the_gpu_note_when_docker_cannot_see_the_gpu(
    server_mode: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        server, "gpu_note", lambda info, **k: "NVIDIA GPU found, but Docker can't use it"
    )
    assert "can't use it" in runner.invoke(cli.app, ["up"]).output
